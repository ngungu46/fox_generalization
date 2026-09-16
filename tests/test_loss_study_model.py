"""Causal format routing and paired gate interventions for the objective study."""

from types import SimpleNamespace
import unittest

import torch

from fox_experiments.loss_study.config import LossStudyConfig
from fox_experiments.loss_study.model import build_model, model_diagnostics


class LossStudyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.data = SimpleNamespace(
            vocab_size=13,
            key_ids=(3, 4, 5, 6),
            value_ids=(7, 8),
            query_id=2,
            bos_id=1,
            pad_id=0,
        )
        self.config = LossStudyConfig.for_profile("smoke")
        self.tokens = torch.tensor([[1, 3, 7, 4, 8, 2, 3], [1, 4, 8, 3, 7, 2, 4]])

    def model(self, gate="factorized_constant", **changes):
        from dataclasses import replace

        return build_model(replace(self.config, **changes), self.data, gate, seed=17)

    def test_constant_parameterizations_and_nongate_weights_match(self):
        factorized, direct = self.model(), self.model("direct_constant")
        torch.testing.assert_close(
            factorized(self.tokens), direct(self.tokens), rtol=0, atol=0
        )
        first, other = dict(factorized.named_parameters()), dict(
            direct.named_parameters()
        )
        for name in first:
            if not name.startswith("blocks.0.attn.gate."):
                torch.testing.assert_close(first[name], other[name], rtol=0, atol=0)
        gate = factorized.blocks[0].attn.gate
        self.assertTrue((gate.u > 0).all())
        torch.testing.assert_close(gate.u, gate.v, rtol=0, atol=0)
        self.assertEqual(direct.config.gate_mode, "direct_constant")

    def test_masks_select_format_without_correct_key_or_future(self):
        model = self.model()
        first = model.format_attention_mask(self.tokens, layer=0)
        # Value at position4 can see both the current and previous KEY tokens.
        self.assertEqual(first[0, 4].nonzero().flatten().tolist(), [1, 3])
        self.assertEqual(first[0, 2].nonzero().flatten().tolist(), [1])
        # Non-value query rows use the same max-backward-distance3 window.
        self.assertEqual(first[0, 6].nonzero().flatten().tolist(), [3, 4, 5, 6])
        later = model.format_attention_mask(self.tokens, layer=1)
        # Query KEY3 sees both records, including the record with irrelevant KEY4.
        self.assertEqual(later[0, 6].nonzero().flatten().tolist(), [2, 4])
        self.assertEqual(later[0, 1].nonzero().flatten().tolist(), [1])
        self.assertFalse(first.triu(1).any())
        self.assertFalse(later.triu(1).any())
        self.assertTrue(first.any(-1).all())
        self.assertTrue(later.any(-1).all())

    def test_missing_key_and_padding_have_finite_self_fallback(self):
        model = self.model()
        tokens = torch.tensor([[7, 0, 0]])
        first = model.format_attention_mask(tokens, layer=0)
        self.assertEqual(first[0, 0].nonzero().flatten().tolist(), [0])
        self.assertTrue(first.any(-1).all())
        self.assertTrue(torch.isfinite(model(tokens)).all())

    def test_full_causal_intervention_and_prefix_causality(self):
        altered = self.tokens.clone()
        altered[:, 4:] = torch.tensor([5, 7, 8])
        for routing in ("record", "causal"):
            model = self.model(routing=routing)
            torch.testing.assert_close(
                model(self.tokens)[:, :4], model(altered)[:, :4], rtol=0, atol=0
            )
            torch.testing.assert_close(
                model(self.tokens),
                model(self.tokens, query_chunk=2),
                rtol=2e-5,
                atol=2e-6,
            )
        causal = self.model(routing="causal")
        for layer in (0, 1):
            self.assertEqual(
                causal.format_attention_mask(self.tokens, layer)[0, 6]
                .nonzero()
                .flatten()
                .tolist(),
                list(range(7)),
            )

    def test_binding_factor_and_retrieval_gate_receive_answer_gradients(self):
        model = self.model()
        labels = torch.full_like(self.tokens, -100)
        labels[:, -1] = torch.tensor([7, 8])
        model.loss(self.tokens, labels, query_chunk=3, logit_chunk=3).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in model.blocks[0].attn.gate.parameters():
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        later = model.blocks[1].attn.gate
        self.assertEqual(later.mode, "direct_constant")
        self.assertEqual(set(dict(later.named_parameters())), {"b"})
        self.assertGreater(float(later.b.grad.abs().sum()), 0)
        x = torch.randn(2, 7, self.config.d_model)
        torch.testing.assert_close(
            later(x), torch.full((2, 2, 7), self.config.retrieval_g0)
        )

    def test_qk_normalization_and_output_gate_ablations(self):
        ordinary = self.model()
        normalized = self.model(qk_norm=True)
        x = normalized.blocks[0].attn_norm(normalized.embedding(self.tokens))
        q, k, _ = normalized.blocks[0].attn.projected(x)
        torch.testing.assert_close(q.norm(dim=-1), torch.ones(2, 2, 7))
        torch.testing.assert_close(k.norm(dim=-1), torch.ones(2, 2, 7))
        gated = self.model(output_gate=True)
        gate = gated.blocks[0].attn.output_gate
        torch.testing.assert_close(gate(x).sigmoid(), torch.full((2, 7, 2), 0.5))
        torch.testing.assert_close(
            gated.blocks[0].attn(x, self.tokens),
            0.5 * ordinary.blocks[0].attn(x, self.tokens),
        )
        gated.loss(self.tokens, (self.tokens + 1) % 13).backward()
        self.assertGreater(float(gate.weight.grad.abs().sum()), 0)

    def test_data_retrieval_and_diagnostics_respect_actual_routing(self):
        model = self.model(retrieval_gate="data")
        self.assertEqual(model.blocks[1].attn.gate.mode, "original_data")
        before = model(self.tokens).detach().clone()
        first = model.query_diagnostics(self.tokens, [2, 2], [4, 4], layer=0)
        self.assertTrue((first["target_attention"] == 0).all())
        later = model.query_diagnostics(self.tokens, [2, 2], [4, 4])
        self.assertTrue(later["target_visible"].all())
        self.assertTrue((later["target_attention"] > 0).all())
        self.assertTrue((later["target_attention"] < 1).all())
        torch.testing.assert_close(before, model(self.tokens), rtol=0, atol=0)
        scalars = model_diagnostics(model, {"input_ids": self.tokens})
        self.assertEqual(scalars["binding_factor_balance_max"], 0)
        self.assertIn("retrieval_gate_mean", scalars)
        self.assertTrue(
            all(torch.isfinite(torch.tensor(value)) for value in scalars.values())
        )

    def test_causal_text_metadata_retains_bos_when_it_equals_padding(self):
        from dataclasses import replace

        data = SimpleNamespace(
            vocab_size=13,
            key_ids=(3, 4, 5, 6),
            value_ids=(7, 8),
            query_id=None,
            bos_id=0,
            pad_id=0,
        )
        config = replace(self.config, data_mode="text_background", routing="causal")
        model = build_model(config, data, "factorized_constant", seed=0)
        tokens = torch.tensor([[0, 3, 7, 0, 2, 4]])
        mask = model.format_attention_mask(tokens, layer=0)
        self.assertIsNone(model.token_format["query_id"])
        self.assertTrue(mask[0, -1].all())
        self.assertTrue(torch.isfinite(model(tokens)).all())


if __name__ == "__main__":
    unittest.main()
