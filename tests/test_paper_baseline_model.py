"""Independent checks of the downscaled upstream-default FoX architecture."""

import math
import hashlib
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from fox_experiments.models import FoXLM, ModelConfig
from fox_experiments.paper_baseline.model import build_comparison_fox, build_paper_fox


def non_gate_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if ".attn.gate." not in name:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class PaperBaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_upstream_defaults_and_analytic_parameter_count(self):
        width, heads, layers, vocab = 128, 2, 3, 101
        model = build_paper_fox(width, layers, heads, vocab)
        self.assertIsInstance(model, FoXLM)
        self.assertEqual(model.config.ff_hidden, 512)
        self.assertEqual(model.config.norm_eps, 1e-6)
        self.assertEqual(model.config.dropout, 0.0)
        self.assertEqual(model.config.gate_mode, "original_data")
        self.assertEqual(model.config.g0, math.log(2.0))
        self.assertEqual(model.config.other_g0, math.log(2.0))
        self.assertNotEqual(
            model.embedding.weight.data_ptr(), model.output_head.weight.data_ptr()
        )
        # Two embeddings; each block has Q/K/V/O, three FF matrices, the
        # head-wise gate matrix/bias, and two RMSNorm vectors; final RMSNorm.
        expected = (
            2 * vocab * width
            + layers
            * (4 * width**2 + 3 * width * 512 + heads * width + heads + 2 * width)
            + width
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
        for block in model.blocks:
            self.assertEqual(block.attn.gate.mode, "original_data")
            self.assertEqual(block.attn_norm.eps, 1e-6)
            self.assertEqual(block.ff_norm.eps, 1e-6)
            self.assertEqual(tuple(block.ff.w1.weight.shape), (512, width))
            self.assertEqual(tuple(block.attn.qkv.weight.shape), (3 * width, width))
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertIsNone(module.bias)

    def test_every_layer_zero_bias_and_data_dependent_gate(self):
        model = build_paper_fox(d_model=32, n_layers=3, n_heads=2, vocab_size=43)
        x = torch.linspace(-2, 2, 2 * 7 * 32).reshape(2, 7, 32)
        for block in model.blocks:
            gate = block.attn.gate
            torch.testing.assert_close(gate.b, torch.zeros(2), rtol=0, atol=0)
            zero_input_decay = gate(torch.zeros_like(x))
            torch.testing.assert_close(
                zero_input_decay, torch.full((2, 2, 7), math.log(2.0))
            )
            self.assertGreater(float(gate.weight.detach().abs().sum()), 0)
            self.assertGreater(float(gate(x).detach().std()), 0.001)

    def test_retention_logsigmoid_sign_and_attention_equivalence(self):
        model = build_paper_fox(d_model=16, n_layers=1, n_heads=2, vocab_size=43)
        attn = model.blocks[0].attn
        with torch.no_grad():
            attn.gate.weight.zero_()
            attn.gate.weight[:, 0].copy_(torch.tensor([0.3, -0.2]))
            attn.gate.b.copy_(torch.tensor([-0.7, 0.4]))
        x = torch.linspace(-1.5, 1.5, 2 * 5 * 16).reshape(2, 5, 16)
        # Write upstream's retention equation directly, with sign-flipped
        # parameters, rather than using the local forgetting-cost helper.
        retention_logits = F.linear(x, -attn.gate.weight, -attn.gate.b)
        log_retention = F.logsigmoid(retention_logits).transpose(1, 2)
        torch.testing.assert_close(log_retention, -attn.gate(x))
        q, k, v = attn.projected(x)
        logits = q @ k.transpose(-1, -2) / math.sqrt(8)
        for query in range(5):
            for key in range(query + 1):
                logits[:, :, query, key] += log_retention[
                    :, :, key + 1 : query + 1
                ].sum(-1)
        logits = logits.masked_fill(
            torch.ones(5, 5, dtype=torch.bool).triu(1), float("-inf")
        )
        manual = attn.out(
            (logits.softmax(-1) @ v).transpose(1, 2).contiguous().view_as(x)
        )
        actual = attn(x, query_chunk=2)
        torch.testing.assert_close(actual, manual, rtol=2e-5, atol=2e-7)
        expected_grad = torch.autograd.grad(
            manual.square().sum(), attn.gate.weight, retain_graph=True
        )[0]
        actual_grad = torch.autograd.grad(actual.square().sum(), attn.gate.weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-5, atol=1e-8)

    def test_projection_initialization_has_no_depth_scaling(self):
        model = build_paper_fox(d_model=128, n_layers=4, n_heads=2, vocab_size=257)
        # These matrices have >=16k samples each: this generous variance
        # check reliably distinguishes std .02 from the old depth-scaled .007.
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and parameter.numel() >= 16000:
                with self.subTest(parameter=name):
                    self.assertLess(abs(float(parameter.detach().std()) - 0.02), 0.0015)
                    self.assertLess(abs(float(parameter.detach().mean())), 0.001)

    def test_seed_pairing_preserves_cpu_rng_and_old_defaults(self):
        torch.manual_seed(713)
        before = torch.random.get_rng_state().clone()
        first = build_paper_fox(d_model=16, vocab_size=43, seed=21)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = build_paper_fox(d_model=16, vocab_size=43, seed=21)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
        old = ModelConfig()
        self.assertEqual(old.gate_mode, "factorized_constant")
        self.assertEqual(old.g0, 0.1)
        self.assertEqual(old.other_g0, 0.1)
        self.assertEqual(old.norm_eps, 1e-5)

    def test_runtime_options_and_trainable_checkpointed_model(self):
        model = build_paper_fox(
            d_model=16,
            n_layers=2,
            n_heads=2,
            vocab_size=43,
            gradient_checkpointing=True,
        )
        self.assertTrue(model.config.gradient_checkpointing)
        tokens = torch.arange(14).reshape(2, 7)
        targets = (tokens + 1) % 43
        loss = model.loss(tokens, targets, query_chunk=3, logit_chunk=4)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        upstream = build_paper_fox(
            d_model=16, vocab_size=43, attention_backend="upstream"
        )
        self.assertEqual(upstream.config.attention_backend, "upstream")
        self.assertTrue(
            all(b.attn.attention_backend == "upstream" for b in upstream.blocks)
        )

    def test_comparison_arms_share_backbone_and_preserve_original(self):
        source = build_paper_fox(d_model=16, n_layers=3, n_heads=2, vocab_size=43)
        before = {name: value.clone() for name, value in source.state_dict().items()}
        tokens = torch.arange(22).reshape(2, 11)
        original_logits = source(tokens).detach().clone()
        backbone_hash = non_gate_hash(source)
        for variant in ("original_data", "factorized_constant", "direct_constant"):
            result = build_comparison_fox(variant, source)
            self.assertEqual(non_gate_hash(result), backbone_hash)
            self.assertNotEqual(
                result.embedding.weight.data_ptr(), source.embedding.weight.data_ptr()
            )
            self.assertEqual(result(tokens).shape, (2, 11, 43))
            if variant == "original_data":
                self.assertEqual(result.config, source.config)
                torch.testing.assert_close(
                    result(tokens), original_logits, rtol=0, atol=0
                )
                for name, value in result.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            else:
                self.assertFalse(torch.allclose(result(tokens), original_logits))
            for name, value in source.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        torch.testing.assert_close(source(tokens), original_logits, rtol=0, atol=0)

    def test_constant_comparison_functions_factors_and_later_gate_settings(self):
        source = build_paper_fox(d_model=16, n_layers=3, n_heads=2, vocab_size=43)
        factorized = build_comparison_fox("factorized_constant", source)
        direct = build_comparison_fox("direct_constant", source)
        gate = factorized.blocks[0].attn.gate
        torch.testing.assert_close(gate.u, gate.v, rtol=0, atol=0)
        expected_factor = math.sqrt(math.log(math.expm1(2.944102615)))
        torch.testing.assert_close(gate.u, torch.full((2,), expected_factor))
        self.assertTrue((gate.u > 0).all())
        torch.testing.assert_close(
            direct.blocks[0].attn.gate.b, gate.u * gate.v, rtol=0, atol=0
        )
        tokens = torch.arange(22).reshape(2, 11)
        torch.testing.assert_close(factorized(tokens), direct(tokens), rtol=0, atol=0)
        arbitrary_inputs = torch.linspace(-3, 3, 2 * 11 * 16).reshape(2, 11, 16)
        torch.testing.assert_close(
            gate(arbitrary_inputs), torch.full((2, 2, 11), 2.944102615)
        )
        for result in (factorized, direct):
            self.assertEqual(result.config.g0, 2.944102615)
            self.assertEqual(result.config.other_g0, 0.1)
            self.assertEqual(result.config.gate_mode, result.blocks[0].attn.gate.mode)
            for index, block in enumerate(result.blocks[1:], start=1):
                torch.testing.assert_close(
                    block.attn.gate.weight,
                    source.blocks[index].attn.gate.weight,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    block.attn.gate(torch.zeros_like(arbitrary_inputs)),
                    torch.full((2, 2, 11), 0.1),
                )
                self.assertEqual(block.attn.gate.mode, "original_data")

    def test_comparison_metadata_recreates_loadable_shapes_and_gradients(self):
        source = build_paper_fox(d_model=16, n_layers=2, n_heads=2, vocab_size=43)
        tokens = torch.arange(14).reshape(2, 7)
        targets = (tokens + 1) % 43
        for variant in ("original_data", "factorized_constant", "direct_constant"):
            result = build_comparison_fox(variant, source, first_g0=1.2, other_g0=0.2)
            restored = FoXLM(result.config)
            restored.load_state_dict(result.state_dict(), strict=True)
            torch.testing.assert_close(restored(tokens), result(tokens), rtol=0, atol=0)
            result.loss(tokens, targets, logit_chunk=4).backward()
            for name, parameter in result.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_comparison_rejects_invalid_constant_initial_region(self):
        source = build_paper_fox(d_model=16, vocab_size=43)
        for variant in ("direct_constant", "factorized_constant"):
            for first_g0 in (0.1, math.log(2), 0.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    build_comparison_fox(variant, source, first_g0=first_g0)
            for other_g0 in (0.0, -1.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    build_comparison_fox(variant, source, other_g0=other_g0)
        with self.assertRaises(ValueError):
            build_comparison_fox("factorized_data", source)
        with self.assertRaises(ValueError):
            build_comparison_fox("direct_constant", FoXLM(ModelConfig(d_model=16)))


if __name__ == "__main__":
    unittest.main()
