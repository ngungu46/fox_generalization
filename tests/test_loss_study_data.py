"""Causal targets, paired histories and counterfactual semantics for loss study."""

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fox_experiments.data import Corpus
from fox_experiments.loss_study.config import LossStudyConfig
from fox_experiments.loss_study.data import StudyData, collate


def config(**kwargs):
    return replace(LossStudyConfig.for_profile("smoke"), **kwargs)


def assert_latest(example, data):
    records = example.metadata["records"]
    key = example.metadata["query_key"]
    matches = [i for i, (k, _) in enumerate(records) if k == key]
    target = matches[-1]
    assert example.tokens[example.answer_index] == data.value_ids[records[target][1]]
    assert example.tokens[example.target_position] == example.tokens[-1]
    assert example.actual_lag == len(records) - target
    assert example.metadata["actual_prefix"] == target
    assert example.target_position == example.metadata["value_positions"][target]
    assert example.metadata["stale_count"] == len(matches) - 1
    assert example.metadata["conflicting_stale_count"] == sum(
        records[i][1] != records[target][1] for i in matches[:-1]
    )
    assert (
        example.metadata["token_lag_query"]
        == example.answer_index - 1 - example.target_position
    )
    assert all(k != key for k, _ in records[target + 1 :])
    if len(matches) > 1:
        assert (
            example.conflict_position
            == example.metadata["value_positions"][matches[-2]]
        )
        assert example.conflict_position < example.target_position
    else:
        assert example.conflict_position == -1


def test_causal_shift_right_padding_and_single_answer_mask():
    data = StudyData(config(batch_size=16))
    examples = data.batch(0, 42)["examples"]
    assert len({len(e.tokens) for e in examples}) > 1
    batch = data.collate(examples)
    assert batch["answer_mask"].sum(dim=1).tolist() == [1] * len(examples)
    for i, example in enumerate(examples):
        n = len(example.tokens) - 1
        assert batch["input_ids"][i, :n].tolist() == example.tokens[:-1]
        assert batch["targets"][i, :n].tolist() == example.tokens[1:]
        assert batch["targets"][i, n:].eq(-100).all()
        assert batch["input_ids"][i, n:].eq(data.pad_id).all()
        assert batch["answer_mask"][i, example.answer_index - 1]
        assert not batch["answer_mask"][i, n:].any()
        assert batch["attention_mask"][i].sum() == n
        assert example.tokens[0] == data.bos_id
        assert example.tokens[-3] == data.query_id
        assert len(example.tokens) == 2 * len(example.metadata["records"]) + 4
        assert_latest(example, data)


def test_same_seed_step_pairs_across_all_arm_choices():
    first = StudyData(config(objectives=("answer_only",), optimizers=("sgd",)))
    second = StudyData(
        config(
            objectives=("all_tokens",),
            optimizers=("adam_fixed",),
            gate_modes=("direct_constant",),
        )
    )
    assert first.fingerprint() == second.fingerprint()
    for step in (0, 1, 71):
        a, b = first.batch(step, 19), second.batch(step, 19)
        for name in ("input_ids", "targets", "answer_mask"):
            torch.testing.assert_close(a[name], b[name])
        assert a["examples"] == b["examples"]
    assert first.batch(0, 19)["examples"] != first.batch(1, 19)["examples"]
    assert first.batch(0, 19)["examples"] != first.batch(0, 20)["examples"]


def test_diverse_support_varies_targets_lags_counts_and_independent_values():
    data = StudyData(
        config(
            batch_size=64,
            n_values=4,
            train_max_lag=4,
            train_max_prefix=4,
            eval_lags=(1, 2, 4),
        )
    )
    examples = [e for step in range(12) for e in data.batch(step, 7)["examples"]]
    assert {e.template for e in examples} == {"calibration", "overwrite", "recall"}
    assert {e.lag for e in examples} == {1, 2, 3, 4}
    assert {e.prefix for e in examples} == {0, 1, 2, 3, 4}
    assert len({len(e.tokens) for e in examples}) >= 6
    by_stale = {value: set() for value in range(4)}
    for example in examples:
        assert_latest(example, data)
        assert example.metadata["target_record"] == example.prefix
        if example.template == "calibration":
            assert len(example.metadata["records"]) == 1
        if example.template == "overwrite":
            record = example.metadata["records"][example.metadata["conflict_record"]]
            answer = example.metadata["answer_value"]
            assert record[1] != answer
            by_stale[record[1]].add(answer)
        else:
            assert example.conflict_position == -1
    assert all(len(answers) >= 2 for answers in by_stale.values())


def test_zero_prefix_training_does_not_sneak_in_an_anchor():
    data = StudyData(config(train_max_prefix=0, batch_size=100))
    examples = data.batch(0, 1)["examples"]
    assert any(e.template == "overwrite_unavailable" for e in examples)
    assert all(e.prefix == 0 and e.conflict_position == -1 for e in examples)


def test_prefix_extension_preserves_core_and_exact_record_count():
    data = StudyData(config(eval_lags=(1, 2, 4), eval_prefixes=(0, 1, 4)))
    bases = [e for e in data.evaluation_examples("long", 82) if e.edit == "base"]
    for example in bases:
        assert_latest(example, data)
        assert len(example.metadata["records"]) == example.prefix + example.lag
        assert example.metadata["target_record"] == example.prefix
        zero = next(
            e
            for e in bases
            if e.lag == example.lag
            and e.prefix == 0
            and e.history_id == example.history_id
        )
        assert example.metadata["records"][example.prefix :] == zero.metadata["records"]
        assert example.tokens[1 + 2 * example.prefix :] == zero.tokens[1:]
        if example.prefix == 0:
            assert example.conflict_position == -1
        else:
            assert example.conflict_position >= 0


def test_each_applicable_edit_is_nonnoop_and_has_correct_latest_answer():
    data = StudyData(config(eval_prefixes=(0, 2), eval_histories=5))
    examples = data.evaluation_examples("long", 88)
    bases = {(e.lag, e.prefix, e.history_id): e for e in examples if e.edit == "base"}
    observed = set()
    for example in examples:
        assert_latest(example, data)
        base = bases[(example.lag, example.prefix, example.history_id)]
        observed.add(example.edit)
        assert len(example.tokens) == len(base.tokens)
        assert example.metadata["base_lag"] == base.lag
        if example.edit == "base":
            continue
        assert example.tokens[:-1] != base.tokens[:-1]
        if example.edit == "latest":
            assert example.tokens[-1] != base.tokens[-1]
        if example.edit in ("stale", "irrelevant"):
            assert example.tokens[-1] == base.tokens[-1]
            assert example.target_position == base.target_position
        if example.edit == "query":
            assert example.metadata["query_key"] != base.metadata["query_key"]
    assert observed == {"base", "latest", "stale", "irrelevant", "query"}
    singleton_edits = {e.edit for e in examples if e.lag == 1 and e.prefix == 0}
    assert singleton_edits == {"base", "latest"}
    assert any(e.actual_lag != e.lag for e in examples if e.edit == "query")
    assert all(e.actual_lag <= e.lag for e in examples if e.edit == "query")
    assert all(
        e.metadata["actual_prefix"] <= max(data.config.eval_prefixes)
        for e in examples
        if e.edit == "query"
    )


def test_lag_panels_append_nested_suffixes_to_the_same_target_and_old_prefix():
    data = StudyData(
        config(eval_lags=(1, 2, 4, 8), eval_prefixes=(0, 1, 4), eval_histories=8)
    )
    examples = [e for e in data.evaluation_examples("long", 41) if e.edit == "base"]
    lookup = {(e.lag, e.prefix, e.history_id): e for e in examples}
    for example in examples:
        longest = lookup[(8, example.prefix, example.history_id)]
        assert example.metadata["query_key"] == longest.metadata["query_key"]
        assert example.metadata["answer_value"] == longest.metadata["answer_value"]
        assert (
            example.metadata["records"]
            == longest.metadata["records"][: example.prefix + example.lag]
        )
        assert (
            example.tokens[1:-3]
            == longest.tokens[1 : 1 + 2 * (example.prefix + example.lag)]
        )
        assert example.tokens[-3:] == longest.tokens[-3:]


def test_stale_density_grows_with_prefix_without_forcing_all_values_to_complement():
    for fraction in (0.0, 1.0):
        data = StudyData(
            config(
                older_stale_fraction=fraction,
                eval_prefixes=(0, 1, 16),
                eval_histories=12,
            )
        )
        examples = [e for e in data.evaluation_examples("long", 4) if e.edit == "base"]
        for example in examples:
            assert_latest(example, data)
            expected = example.prefix if fraction == 1 else int(example.prefix > 0)
            assert example.metadata["stale_count"] == expected
            if example.prefix:
                assert example.metadata["conflicting_stale_count"] >= 1
        if fraction == 1:
            long = [e for e in examples if e.prefix == 16]
            assert all(
                e.metadata["conflicting_stale_count"] < e.metadata["stale_count"]
                for e in long
            )
            # Only the anchor is forced to disagree; old values may match either
            # answer, and carry no deterministic answer-complement rule.
            pairs = {
                (e.metadata["records"][0][1], e.metadata["answer_value"]) for e in long
            }
            assert pairs == {(0, 0), (0, 1), (1, 0), (1, 1)}
    half = StudyData(
        config(older_stale_fraction=0.5, eval_prefixes=(0, 1, 16), eval_histories=32)
    )
    long = [
        e
        for e in half.evaluation_examples("long", 8)
        if e.edit == "base" and e.prefix == 16
    ]
    assert np.mean([e.metadata["stale_count"] for e in long]) > 5


def test_short_and_tune_actual_geometry_is_inside_training_envelope():
    cfg = config(train_max_lag=2, train_max_prefix=1, eval_prefixes=(0, 1, 8))
    data = StudyData(cfg)
    for split in ("short", "tune"):
        examples = data.evaluation_examples(split, 3)
        assert {e.lag for e in examples} == {1, 2}
        assert {e.prefix for e in examples} == {0, 1}
        assert all(e.actual_lag <= 2 for e in examples)
        assert all(e.metadata["actual_prefix"] <= 1 for e in examples)
    assert data.evaluation_examples("short", 3) == data.evaluation_examples("short", 3)
    assert data.evaluation_examples("short", 3) != data.evaluation_examples("tune", 3)
    assert data.evaluation_examples("short", 3) != data.evaluation_examples("long", 3)


def test_theory_templates_are_literal_and_have_no_implicit_recall_extension():
    cfg = config(
        train_support="theory_templates",
        train_max_prefix=3,
        train_max_lag=2,
        batch_size=120,
    )
    data = StudyData(cfg)
    examples = data.batch(0, 17)["examples"]
    assert {e.template for e in examples} == {"calibration", "overwrite", "recall"}
    for example in examples:
        assert_latest(example, data)
        records = example.metadata["records"]
        a, positive = example.metadata["query_key"], example.metadata["answer_value"]
        negative = 1 - positive
        if example.template == "calibration":
            assert records == [[a, positive]]
        elif example.template == "overwrite":
            b = records[2][0]
            assert b != a
            assert records == [
                [a, negative],
                [a, negative],
                [b, negative],
                [a, positive],
            ]
            assert example.lag == 1 and example.prefix == 3
        else:
            assert records[0] == [a, positive]
            assert records[1][0] != a and records[1][1] == negative
            assert example.lag == 2 and len(records) == 2
    with pytest.raises(ValueError, match="Literal theory"):
        StudyData(config(train_support="theory_templates", train_max_prefix=1))


def test_text_wrapper_has_single_token_answers_and_document_local_provenance():
    cache = (
        Path(__file__).resolve().parents[1] / "data/longcrawl64_pilot/tokenizer_cache"
    )
    if not cache.is_dir():
        pytest.skip("Offline GPT-2 cache is not available")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
    # Procedural source arrays are for software validation only.
    train = np.tile(np.arange(1000, 1128, dtype=np.uint16), (4, 1))
    validation = np.tile(np.arange(1200, 1328, dtype=np.uint16), (4, 1))
    test = np.tile(np.arange(1400, 1528, dtype=np.uint16), (4, 1))
    cfg = config(data_mode="text_background", routing="causal", batch_size=12)
    data = StudyData(cfg, corpus=Corpus(train, validation, test))
    assert data.vocab_size == 50257 and data.query_id is None
    assert data.fingerprint()["corpus_sha256"]
    for split in ("short", "long", "tune"):
        examples = data.evaluation_examples(split, 12)
        for example in examples:
            assert_latest(example, data)
            for snippet in example.metadata["source_snippets"]:
                assert snippet["length"] == 8
                assert snippet["offset"] + 8 <= 128
                assert (
                    snippet["split"]
                    == {"short": "confirm", "long": "test", "tune": "tune"}[split]
                )
            assert "Update: key " in data.render_example(example)
            assert data.tokenizer.decode([example.tokens[-1]]) in ("0", "1")
    batch = data.batch(0, 6)
    assert batch["answer_mask"].sum() == cfg.batch_size
    assert batch["input_ids"][~batch["attention_mask"]].eq(data.pad_id).all()

    bases = [e for e in data.evaluation_examples("long", 31) if e.edit == "base"]
    lookup = {(e.lag, e.prefix, e.history_id): e for e in bases}
    header_length = len(data.tokenizer.encode("\nUpdate: key "))
    trailer_length = len(data.tokenizer.encode(".\n"))

    def record_tokens(example, index):
        start = example.metadata["key_positions"][index] - header_length - 8
        stop = example.metadata["value_positions"][index] + 1 + trailer_length
        return example.tokens[start:stop]

    for example in bases:
        zero = lookup[(example.lag, 0, example.history_id)]
        for offset in range(example.lag):
            assert record_tokens(example, example.prefix + offset) == record_tokens(
                zero, offset
            )
            assert (
                example.metadata["source_snippets"][example.prefix + offset]
                == zero.metadata["source_snippets"][offset]
            )
        longest = lookup[(max(cfg.eval_lags), example.prefix, example.history_id)]
        for index in range(example.prefix + example.lag):
            assert record_tokens(example, index) == record_tokens(longest, index)
            assert (
                example.metadata["source_snippets"][index]
                == longest.metadata["source_snippets"][index]
            )
    for edited in data.evaluation_examples("long", 31):
        base = lookup[(edited.lag, edited.prefix, edited.history_id)]
        assert edited.metadata["source_snippets"] == base.metadata["source_snippets"]


def test_empty_collation_and_invalid_split_fail_explicitly():
    with pytest.raises(ValueError, match="empty"):
        collate([])
    with pytest.raises(ValueError, match="Evaluation split"):
        StudyData(config()).evaluation_examples("validation")
