"""Meaningful numerical checks for the FoX kernel and paired interventions."""

import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from fox_experiments.models import ModelConfig, FoXLM, forgetting_cost, per_token_nll


def tiny(mode="factorized_constant", **kwargs):
    return FoXLM(
        ModelConfig(
            vocab_size=43,
            d_model=24,
            n_heads=3,
            n_layers=2,
            ff_hidden=32,
            gate_mode=mode,
            **kwargs,
        )
    )


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.tokens = torch.randint(0, 43, (2, 11))

    def test_prefix_cost_excludes_key_includes_query(self):
        rates = torch.tensor([[[0.2, 0.4, 0.8, 1.6]]])
        expected = torch.tensor(
            [
                [0.0, -0.4, -1.2, -2.8],
                [0.4, 0.0, -0.8, -2.4],
                [1.2, 0.8, 0.0, -1.6],
                [2.8, 2.4, 1.6, 0.0],
            ]
        )
        torch.testing.assert_close(forgetting_cost(rates)[0, 0], expected)
        torch.testing.assert_close(forgetting_cost(rates, 1, 3)[0, 0], expected[1:3])

    def test_dense_chunk_and_manual_attention(self):
        model = tiny().eval()
        dense, chunked = model(self.tokens), model(self.tokens, query_chunk=3)
        torch.testing.assert_close(dense, chunked, rtol=2e-5, atol=2e-6)
        attn = model.blocks[0].attn
        x = model.blocks[0].attn_norm(model.embedding(self.tokens))
        q, k, v = attn.projected(x)
        scores = q @ k.transpose(-1, -2) / (attn.d_head**0.5)
        scores -= forgetting_cost(attn.gate(x))
        scores.masked_fill_(torch.ones(11, 11, dtype=torch.bool).triu(1), float("-inf"))
        manual = (scores.softmax(-1) @ v).transpose(1, 2).contiguous().view_as(x)
        torch.testing.assert_close(
            attn(x, query_chunk=4), attn.out(manual), rtol=2e-5, atol=2e-6
        )

    def test_causality_all_gate_modes(self):
        changed = self.tokens.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 43
        for mode in (
            "direct_constant",
            "factorized_constant",
            "original_data",
            "factorized_data",
        ):
            model = tiny(mode).eval()
            torch.testing.assert_close(
                model(self.tokens)[:, :6], model(changed)[:, :6], rtol=1e-5, atol=1e-6
            )

    def test_function_matched_init_and_other_parameters(self):
        for direct, factorized in (
            ("direct_constant", "factorized_constant"),
            ("original_data", "factorized_data"),
        ):
            a, b = tiny(direct), tiny(factorized)
            torch.testing.assert_close(a(self.tokens), b(self.tokens), rtol=0, atol=0)
            aparams, bparams = dict(a.named_parameters()), dict(b.named_parameters())
            for name in aparams:
                if not name.startswith("blocks.0.attn.gate."):
                    torch.testing.assert_close(
                        aparams[name], bparams[name], rtol=0, atol=0
                    )

    def test_conversion_after_arbitrary_checkpoint(self):
        for direct, factorized in (
            ("direct_constant", "factorized_constant"),
            ("original_data", "factorized_data"),
        ):
            model = tiny(factorized)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.02 * torch.randn_like(parameter))
                model.blocks[0].attn.gate.u.copy_(torch.tensor([-0.7, 0.0, 1.4]))
            converted = model.converted(direct)
            roundtrip = converted.converted(factorized)
            torch.testing.assert_close(
                model(self.tokens), converted(self.tokens), rtol=3e-5, atol=3e-6
            )
            torch.testing.assert_close(
                model(self.tokens), roundtrip(self.tokens), rtol=3e-5, atol=3e-6
            )
            self.assertNotEqual(
                model.embedding.weight.data_ptr(), converted.embedding.weight.data_ptr()
            )
        with self.assertRaises(ValueError):
            tiny().converted("original_data")

    def test_masked_chunk_nll_and_gradients(self):
        targets = torch.randint(0, 43, self.tokens.shape)
        targets[0, 2] = -100
        weights = torch.ones_like(targets, dtype=torch.float32)
        weights[:, -1] = 4
        weights[1, :3] = 0
        for mode in (
            "direct_constant",
            "factorized_constant",
            "original_data",
            "factorized_data",
        ):
            model = tiny(mode)
            loss = model.loss(
                self.tokens, targets, loss_mask=weights, query_chunk=3, logit_chunk=4
            )
            dense_nll = F.cross_entropy(
                model(self.tokens).reshape(-1, 43),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(weights)
            effective = weights * (targets != -100)
            torch.testing.assert_close(
                loss, (dense_nll * effective).sum() / effective.sum()
            )
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(
                sum(p.grad.abs().sum().item() for p in model.gate_parameters(0)), 0
            )

    def test_checkpointed_nll_gradient_matches_dense(self):
        model = tiny()
        other = model.converted("factorized_constant")
        targets = torch.randint(0, 43, self.tokens.shape)
        model.loss(self.tokens, targets, logit_chunk=3).backward()
        F.cross_entropy(
            other(self.tokens).reshape(-1, 43), targets.reshape(-1)
        ).backward()
        for (name, p), (_, q) in zip(
            model.named_parameters(), other.named_parameters()
        ):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_diagnostics_integrated_cost_and_no_forward_sidechannel(self):
        model = tiny(g0=0.17).train()
        before = model(self.tokens)
        d = model.last_query_diagnostics(
            self.tokens, torch.tensor([2, 4]), torch.tensor([8, -1])
        )
        expected = torch.tensor([8, 6])[:, None].expand(-1, 3) * 0.17
        torch.testing.assert_close(d["integrated_target_forgetting_cost"], expected)
        self.assertTrue(model.training)
        self.assertTrue(torch.isnan(d["content_margin_vs_conflict"][1]).all())
        self.assertTrue(
            ((d["target_attention"] > 0) & (d["target_attention"] < 1)).all()
        )
        torch.testing.assert_close(before, model(self.tokens), rtol=0, atol=0)

    def test_autocast_gate_fp32(self):
        model = tiny("original_data")
        x = model.embedding(self.tokens)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertEqual(model.blocks[0].attn.gate(x).dtype, torch.float32)
            y = model(self.tokens, query_chunk=3)
        self.assertTrue(torch.isfinite(y).all())

    def test_strong_local_initialization(self):
        g0 = F.softplus(torch.tensor(1.7**2)).item()
        model = tiny(g0=g0)
        x = model.embedding(self.tokens)
        torch.testing.assert_close(
            model.blocks[0].attn.gate(x), torch.full((2, 3, 11), g0)
        )
        torch.testing.assert_close(model.blocks[0].attn.gate.u, torch.full((3,), 1.7))
        torch.testing.assert_close(model.blocks[0].attn.gate.v, torch.full((3,), 1.7))
        # Changing the first binding gate leaves the retrieval gate unchanged.
        weak = tiny(g0=0.1)
        torch.testing.assert_close(
            model.blocks[1].attn.gate(x), weak.blocks[1].attn.gate(x), rtol=0, atol=0
        )
        direct = tiny("direct_constant", g0=g0)
        balanced = direct.converted("factorized_constant")
        torch.testing.assert_close(
            balanced.blocks[0].attn.gate.u,
            balanced.blocks[0].attn.gate.v,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model(self.tokens), direct(self.tokens), rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            balanced(self.tokens), direct(self.tokens), rtol=2e-5, atol=2e-6
        )

    def test_conversion_mixed_bias_signs(self):
        direct = tiny("direct_constant")
        with torch.no_grad():
            direct.blocks[0].attn.gate.b.copy_(torch.tensor([-1.5, 0.0, 2.89]))
        factorized = direct.converted("factorized_constant")
        gate = factorized.blocks[0].attn.gate
        torch.testing.assert_close(gate.u, torch.tensor([1.0, 1.0, 1.7]))
        torch.testing.assert_close(gate.v, torch.tensor([-1.5, 0.0, 1.7]))
        torch.testing.assert_close(
            factorized(self.tokens), direct(self.tokens), rtol=2e-5, atol=2e-6
        )

    def test_activation_checkpointing_preserves_loss_and_gradients(self):
        reference = tiny("factorized_constant", g0=2.944102615)
        checkpointed = FoXLM(replace(reference.config, gradient_checkpointing=True))
        targets = torch.randint(0, 43, self.tokens.shape)
        expected = reference.loss(self.tokens, targets, query_chunk=3, logit_chunk=4)
        actual = checkpointed.loss(self.tokens, targets, query_chunk=3, logit_chunk=4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.backward()
        actual.backward()
        for (name, left), (_, right) in zip(
            reference.named_parameters(), checkpointed.named_parameters()
        ):
            torch.testing.assert_close(
                left.grad, right.grad, rtol=2e-5, atol=2e-6, msg=name
            )


if __name__ == "__main__":
    unittest.main()
