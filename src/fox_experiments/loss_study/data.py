"""Paired latest-write examples for the objective comparison.

Record histories, random draws and edits do not depend on the optimizer, gate or
loss objective. ``lag`` and ``prefix`` label the original evaluation grid cell;
``actual_lag`` and metadata ``actual_prefix`` describe a changed query. All token
positions refer to the unshifted sequence, including its initial BOS token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

PROTOCOL = "loss-study-latest-write-v2"


def _seed(*parts):
    message = json.dumps(parts, separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little")


def _rng(*parts):
    return np.random.default_rng(_seed(PROTOCOL, *parts))


def _different(rng, value, size):
    draw = int(rng.integers(size - 1))
    return draw + (draw >= value)


@dataclass
class Example:
    tokens: list[int]
    answer_index: int
    target_position: int
    conflict_position: int
    lag: int
    prefix: int
    history_id: int
    edit: str = "base"
    template: str = "recall"
    actual_lag: int = 1
    metadata: dict = field(default_factory=dict)


def collate(examples, device="cpu", pad_id=0):
    """Right-pad inputs and mask every padded target with -100.

    The answer at full-sequence index ``answer_index`` is predicted from input
    index ``answer_index-1``. It appears exactly once in ``answer_mask``.
    """
    examples = list(examples)
    if not examples:
        raise ValueError("Cannot collate an empty example collection")
    length = max(len(example.tokens) - 1 for example in examples)
    input_ids = torch.full((len(examples), length), pad_id, dtype=torch.long)
    targets = torch.full_like(input_ids, -100)
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, example in enumerate(examples):
        n = len(example.tokens) - 1
        if not 1 <= example.answer_index <= n:
            raise ValueError("Answer must be a predicted sequence token")
        input_ids[index, :n] = torch.tensor(example.tokens[:-1])
        targets[index, :n] = torch.tensor(example.tokens[1:])
        answer_mask[index, example.answer_index - 1] = True
        attention_mask[index, :n] = True
    return {
        "input_ids": input_ids.to(device),
        "targets": targets.to(device),
        "answer_mask": answer_mask.to(device),
        "attention_mask": attention_mask.to(device),
        "examples": examples,
    }


class StudyData:
    """Deterministic record tasks with an optional semi-synthetic text wrapper."""

    def __init__(self, config, data_dir=None, corpus=None):
        self.config = config
        self.corpus = corpus
        self.tokenizer = None
        self.source_manifest = None
        if config.train_support == "theory_templates":
            if config.train_max_lag != 2 or config.train_max_prefix != 3:
                raise ValueError(
                    "Literal theory templates require train_max_lag=2 and "
                    "train_max_prefix=3 (overwrite has three prefix records)"
                )
        if config.data_mode == "records":
            self.pad_id, self.bos_id, self.query_id = 0, 1, 2
            self.key_ids = list(range(3, 3 + config.n_keys))
            self.value_ids = list(
                range(3 + config.n_keys, 3 + config.n_keys + config.n_values)
            )
            self.vocab_size = 3 + config.n_keys + config.n_values
        else:
            if config.n_values > 10 or config.n_keys > 52:
                raise ValueError(
                    "Text mode supports at most 10 digits and 52 letter keys"
                )
            import tiktoken
            from fox_experiments.data import Corpus, load_corpus

            if data_dir is not None:
                data_dir = Path(data_dir)
                cache = data_dir / "tokenizer_cache"
                if cache.is_dir():
                    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache.resolve()))
            if self.corpus is None:
                if data_dir is None:
                    raise ValueError("text_background requires data_dir or a Corpus")
                files = load_corpus(data_dir)
                self.source_manifest = files.manifest
                self.corpus = Corpus(files.train, files.validation, files.test)
            self.tokenizer = tiktoken.get_encoding("gpt2")
            self.vocab_size = self.tokenizer.n_vocab
            self.pad_id = self.bos_id = self.tokenizer.eot_token
            self.query_id = None
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            self.key_ids = [
                self._single(letter) for letter in alphabet[: config.n_keys]
            ]
            self.value_ids = [
                self._single(str(value)) for value in range(config.n_values)
            ]
        self._fingerprint = self._make_fingerprint()

    def _single(self, text):
        ids = self.tokenizer.encode(text)
        if len(ids) != 1:
            raise ValueError(f"Expected a single GPT-2 token for {text!r}")
        return ids[0]

    def _make_fingerprint(self):
        fields = (
            "data_mode",
            "train_support",
            "n_keys",
            "n_values",
            "train_max_lag",
            "train_max_prefix",
            "older_stale_fraction",
            "short_lags",
            "eval_lags",
            "eval_prefixes",
            "eval_histories",
            "batch_size",
        )
        vocabulary = {
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "pad_id": self.pad_id,
            "query_id": self.query_id,
            "key_ids": self.key_ids,
            "value_ids": self.value_ids,
        }
        details = {
            "protocol": PROTOCOL,
            "configuration": {name: getattr(self.config, name) for name in fields},
            "vocabulary": vocabulary,
            "vocabulary_sha256": hashlib.sha256(
                json.dumps(vocabulary, sort_keys=True).encode()
            ).hexdigest(),
            "corpus_sha256": None,
        }
        if self.corpus is not None:
            if self.source_manifest is not None:
                details["corpus_sha256"] = self.source_manifest["sha256"]
                details["corpus_source_revision"] = self.source_manifest.get(
                    "source_revision"
                )
            else:
                arrays = getattr(self.corpus, "arrays", None)
                if arrays is None:
                    raise ValueError(
                        "Injected text Corpus must expose its source arrays"
                    )
                digest = hashlib.sha256()
                for name in ("train", "validation", "test"):
                    array = np.asarray(arrays[name])
                    digest.update(name.encode())
                    digest.update(json.dumps(list(array.shape)).encode())
                    digest.update(np.asarray(array, dtype="<u2").tobytes())
                details["corpus_sha256"] = digest.hexdigest()
        details["data_sha256"] = hashlib.sha256(
            json.dumps(details, sort_keys=True).encode()
        ).hexdigest()
        return json.loads(json.dumps(details))

    def fingerprint(self):
        """Stable JSON data identity, excluding paths and experimental arm choices."""
        return json.loads(json.dumps(self._fingerprint))

    def _history(self, namespace, seed, history_id, lag, prefix, overwrite=False):
        # Lag/prefix panels extend one shared history: the target stays fixed,
        # and longer lags append the same nested sequence of newer off-key writes.
        core = _rng(namespace, seed, history_id, "core")
        query = int(core.integers(self.config.n_keys))
        answer = int(core.integers(self.config.n_values))
        records = [(query, answer)]
        for _ in range(lag - 1):
            records.append(
                (
                    _different(core, query, self.config.n_keys),
                    int(core.integers(self.config.n_values)),
                )
            )
        older = _rng(namespace, seed, history_id, "prefix")
        before = []
        for _ in range(prefix):
            matching = overwrite and older.random() < self.config.older_stale_fraction
            key = query if matching else _different(older, query, self.config.n_keys)
            before.append((key, int(older.integers(self.config.n_values))))
        if overwrite and prefix:
            # Conditional disagreement is explicit; for binary values it is
            # necessarily a complement. Other records remain independent draws.
            before[-1] = (query, _different(older, answer, self.config.n_values))
        return before + records, query

    def _render(
        self,
        records,
        query,
        *,
        lag,
        prefix,
        history_id,
        namespace,
        seed,
        template="recall",
        edit="base",
        inherited=None,
    ):
        matching = [index for index, (key, _) in enumerate(records) if key == query]
        if not matching:
            raise ValueError("Query must refer to a key present in the history")
        target = matching[-1]
        conflict = matching[-2] if len(matching) > 1 else -1
        tokens = [self.bos_id]
        key_positions, value_positions, sources = [], [], []
        for index, (key, value) in enumerate(records):
            if self.tokenizer is not None:
                # This index is relative to the BASE target, including for
                # counterfactual query edits; geometry never resamples background.
                relative_index = index - prefix
                background_rng = _rng(
                    namespace, seed, history_id, "text", relative_index
                )
                source_split = {
                    "train": "train",
                    "tune": "tune",
                    "short": "confirm",
                    "long": "test",
                }[namespace]
                snippet, row, offset = self.corpus.sample(
                    source_split, 8, background_rng
                )
                tokens.extend(snippet)
                tokens.extend(self.tokenizer.encode("\nUpdate: key "))
                sources.append(
                    {"split": source_split, "row": row, "offset": offset, "length": 8}
                )
            key_positions.append(len(tokens))
            tokens.append(self.key_ids[key])
            if self.tokenizer is not None:
                tokens.extend(self.tokenizer.encode(" = "))
            value_positions.append(len(tokens))
            tokens.append(self.value_ids[value])
            if self.tokenizer is not None:
                tokens.extend(self.tokenizer.encode(".\n"))
        if self.tokenizer is None:
            tokens.append(self.query_id)
        else:
            tokens.extend(self.tokenizer.encode("\nQuery: latest value for key "))
        query_position = len(tokens)
        tokens.append(self.key_ids[query])
        if self.tokenizer is not None:
            tokens.extend(self.tokenizer.encode(" = "))
        answer_index = len(tokens)
        tokens.append(self.value_ids[records[target][1]])
        metadata = {
            "records": [list(record) for record in records],
            "query_key": query,
            "answer_value": records[target][1],
            "target_record": target,
            "conflict_record": conflict,
            "stale_count": len(matching) - 1,
            "conflicting_stale_count": sum(
                records[index][1] != records[target][1] for index in matching[:-1]
            ),
            "stale_positions": [value_positions[index] for index in matching[:-1]],
            "key_positions": key_positions,
            "value_positions": value_positions,
            "query_position": query_position,
            "actual_prefix": target,
            "base_lag": lag,
            "base_prefix": prefix,
            "base_history_id": history_id,
            "namespace": namespace,
            "token_lag": answer_index - value_positions[target],
            "token_lag_query": answer_index - 1 - value_positions[target],
            "source_snippets": sources,
        }
        if inherited:
            metadata.update(inherited)
        return Example(
            tokens=tokens,
            answer_index=answer_index,
            target_position=value_positions[target],
            conflict_position=value_positions[conflict] if conflict >= 0 else -1,
            lag=lag,
            prefix=prefix,
            history_id=history_id,
            edit=edit,
            template=template,
            actual_lag=len(records) - target,
            metadata=metadata,
        )

    def _training_example(self, step, index, seed):
        history_id = int(step) * self.config.batch_size + index
        rng = _rng("train", seed, step, index, "template")
        draw = float(rng.random())
        template = (
            "calibration" if draw < 0.6 else ("overwrite" if draw < 0.8 else "recall")
        )
        if self.config.train_support == "theory_templates":
            a = int(rng.integers(self.config.n_keys))
            b = _different(rng, a, self.config.n_keys)
            positive = int(rng.integers(2))
            negative = 1 - positive
            if template == "calibration":
                records, lag, prefix = [(a, positive)], 1, 0
            elif template == "overwrite":
                records = [(a, negative), (a, negative), (b, negative), (a, positive)]
                lag, prefix = 1, 3
            else:
                records, lag, prefix = [(a, positive), (b, negative)], 2, 0
            query = a
        else:
            lag = (
                1
                if template == "calibration"
                else int(rng.integers(1, self.config.train_max_lag + 1))
            )
            if template == "calibration":
                prefix = 0
            elif template == "overwrite" and self.config.train_max_prefix:
                prefix = int(rng.integers(1, self.config.train_max_prefix + 1))
            else:
                prefix = int(rng.integers(self.config.train_max_prefix + 1))
                if template == "overwrite":
                    template = "overwrite_unavailable"
            records, query = self._history(
                "train",
                seed,
                history_id,
                lag,
                prefix,
                overwrite=template == "overwrite",
            )
        return self._render(
            records,
            query,
            lag=lag,
            prefix=prefix,
            history_id=history_id,
            namespace="train",
            seed=seed,
            template=template,
        )

    def batch(self, step, seed, device="cpu"):
        if step < 0:
            raise ValueError("Training step must be nonnegative")
        examples = [
            self._training_example(step, index, seed)
            for index in range(self.config.batch_size)
        ]
        return self.collate(examples, device)

    def collate(self, examples, device="cpu"):
        return collate(examples, device=device, pad_id=self.pad_id)

    def _edits(self, base, seed):
        yield base
        records = [tuple(record) for record in base.metadata["records"]]
        query = base.metadata["query_key"]
        target = base.metadata["target_record"]
        rng = _rng(
            base.metadata["namespace"],
            seed,
            base.history_id,
            base.lag,
            base.prefix,
            "edits",
        )
        changes = []
        latest = records.copy()
        latest[target] = (
            query,
            _different(rng, records[target][1], self.config.n_values),
        )
        changes.append(("latest", latest, query))
        conflict = base.metadata["conflict_record"]
        if conflict >= 0:
            stale = records.copy()
            stale[conflict] = (
                query,
                _different(rng, records[conflict][1], self.config.n_values),
            )
            changes.append(("stale", stale, query))
        irrelevant_positions = [i for i, (key, _) in enumerate(records) if key != query]
        if irrelevant_positions:
            position = irrelevant_positions[-1]
            irrelevant = records.copy()
            key, value = irrelevant[position]
            irrelevant[position] = (key, _different(rng, value, self.config.n_values))
            changes.append(("irrelevant", irrelevant, query))
            available = []
            for alternative in sorted({records[i][0] for i in irrelevant_positions}):
                alternative_target = max(
                    i for i, (key, _) in enumerate(records) if key == alternative
                )
                actual_lag = len(records) - alternative_target
                if base.metadata["namespace"] == "long":
                    eligible = actual_lag <= base.lag and alternative_target <= max(
                        self.config.eval_prefixes
                    )
                else:
                    eligible = (
                        actual_lag <= self.config.train_max_lag
                        and alternative_target <= self.config.train_max_prefix
                    )
                if eligible:
                    available.append(alternative)
            if available:
                alternative = available[int(rng.integers(len(available)))]
                changes.append(("query", records.copy(), alternative))
        for edit, edited_records, edited_query in changes:
            yield self._render(
                edited_records,
                edited_query,
                lag=base.lag,
                prefix=base.prefix,
                history_id=base.history_id,
                namespace=base.metadata["namespace"],
                seed=seed,
                template=base.template,
                edit=edit,
                inherited={
                    "base_answer_value": base.metadata["answer_value"],
                    "base_target_position": base.target_position,
                    "base_stale_count": base.metadata["stale_count"],
                },
            )

    def evaluation_examples(self, split="short", seed=0):
        """Fixed paired histories with only applicable, non-noop interventions.

        Short/learning-rate-tune data stay within the declared training envelope.
        Long data use the complete crossed lag/prefix grid. Evaluation namespaces
        are independent of each other and of training, including text documents.
        """
        if split not in ("short", "long", "tune"):
            raise ValueError("Evaluation split must be 'short', 'long' or 'tune'")
        if split == "long":
            lags, prefixes = self.config.eval_lags, self.config.eval_prefixes
        else:
            lags = self.config.short_lags
            prefixes = sorted(
                {0, self.config.train_max_prefix}
                | {
                    p
                    for p in self.config.eval_prefixes
                    if p <= self.config.train_max_prefix
                }
            )
        examples = []
        for lag in lags:
            for prefix in prefixes:
                for history_id in range(self.config.eval_histories):
                    records, query = self._history(
                        split, seed, history_id, lag, prefix, overwrite=prefix > 0
                    )
                    base = self._render(
                        records,
                        query,
                        lag=lag,
                        prefix=prefix,
                        history_id=history_id,
                        namespace=split,
                        seed=seed,
                        template="overwrite" if prefix else "recall",
                    )
                    for edited in self._edits(base, seed):
                        if split != "long" and (
                            edited.actual_lag > self.config.train_max_lag
                            or edited.metadata["actual_prefix"]
                            > self.config.train_max_prefix
                        ):
                            # A changed query can move the actual target outside
                            # the training envelope despite unchanged panel labels.
                            continue
                        if (
                            split == "long"
                            and edited.edit == "query"
                            and (
                                edited.actual_lag > lag
                                or edited.metadata["actual_prefix"]
                                > max(self.config.eval_prefixes)
                            )
                        ):
                            # Do not let a changed query import a larger actual
                            # recall lag into a smaller base-lag success panel.
                            continue
                        examples.append(edited)
        return examples

    def describe(self):
        return {
            "protocol": PROTOCOL,
            "data_mode": self.config.data_mode,
            "sequence": "BOS, (key,value)*, QUERY, query_key, answer; no EOS",
            "lag": "Target-inclusive record count from latest queried write to final record",
            "prefix": "Exact record count before the base example's target write",
            "training_mixture": {"calibration": 0.6, "overwrite": 0.2, "recall": 0.2},
            "train_support": self.config.train_support,
            "literal_theory_support": "cal=[(a,+)], O=[(a,-),(a,-),(b,-),(a,+)], C=[(a,+),(b,-)]; random key pair and binary complement; no automatic recall extension",
            "diverse_support": "Overwrite/evaluation prefixes independently match the query with older_stale_fraction, with iid values; the last prefix record is forced to match and disagree with the latest value. Thus expected stale count is 1+(prefix-1)*older_stale_fraction when prefix>0. Recall/calibration training has no stale writes. Binary forced-anchor disagreement determines its complement; other stale values remain iid.",
            "zero_prefix_behavior": "If train_max_prefix=0, overwrite draws become explicitly labeled overwrite_unavailable examples without a stale write.",
            "text_background": "Semi-synthetic: each structured update is preceded by an 8-token document-local public-corpus snippet; GPT-2 single-digit answer; no natural-language QA claim.",
            "evaluation": "Independent train/tune/short/long RNG namespaces; lag panels append nested suffixes to the same query/answer and old prefix; prefix extensions preserve their target and suffix. Text snippets use record indices relative to the base target, preserving background across both axes and counterfactual edits. Omit inapplicable edits. Short/tune query edits stay within actual training lag/prefix limits. Long query edits have actual_lag <= base_lag and actual_prefix <= maximum evaluation prefix; base grid labels remain unchanged.",
            "fingerprint": self.fingerprint(),
        }

    def render_example(self, example):
        if self.tokenizer is not None:
            return self.tokenizer.decode(example.tokens)
        record_text = ", ".join(
            f"k{key}=v{value}" for key, value in example.metadata["records"]
        )
        return (
            f"BOS [{record_text}] QUERY k{example.metadata['query_key']} "
            f"ANSWER v{example.metadata['answer_value']} "
            f"(edit={example.edit}, base_lag={example.lag}, actual_lag={example.actual_lag}, prefix={example.prefix})"
        )
