"""Small, explicit bridge to the authors' memory-efficient attention kernel."""

from __future__ import annotations

import torch


def upstream_forgetting_attention(q, k, v, log_fgate):
    """Accept head-first QKV and FP32 log gates, with pruning disabled.

    No dense fallback is hidden here: selecting the full backend requires CUDA
    and FP16/BF16 QKV. Parameters and gate calculations remain FP32 under AMP.
    """
    if q.device.type != "cuda":
        raise RuntimeError("The upstream Triton backend requires an NVIDIA CUDA GPU")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Run the upstream backend inside CUDA BF16 autocast")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("Upstream kernel supports head widths 16, 32, 64, 128")
    from ._vendor.forgetting_attention import forgetting_attention

    return forgetting_attention(
        q,
        k,
        v,
        log_fgate.float().contiguous(),
        head_first=True,
        adaptive_threshold=None,
    )


def check_kernel():
    """One forward/backward comparison against a dense FP32 reference on CUDA.

    This is a required hardware preflight, not an already verified GPU result.
    Gate gradients are checked as well as Q/K/V gradients.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Kernel validation needs a CUDA runtime")
    cases = []
    for length, decay in ((128, 0.03), (137, 2.944102615)):
        cases.append(_check_case(length, decay))
    return {
        "kernel_check": "passed",
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "pruning": False,
        "cases": cases,
    }


def _check_case(length, decay):
    torch.manual_seed(123)
    q, k, v = [
        torch.randn(
            1, 2, length, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(3)
    ]
    gate = torch.full((1, 2, length), -decay, device="cuda", requires_grad=True)
    actual = upstream_forgetting_attention(q, k, v, gate)
    probe = torch.randn_like(actual)
    (actual * probe).float().sum().backward()
    actual_grads = [x.grad.detach().float() for x in (q, k, v, gate)]
    qr, kr, vr, gr = [x.detach().float().requires_grad_() for x in (q, k, v, gate)]
    prefix = gr.cumsum(-1)
    scores = qr @ kr.transpose(-1, -2) / 8 + prefix[..., :, None] - prefix[..., None, :]
    causal = torch.ones(length, length, dtype=torch.bool, device="cuda").tril()
    expected = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ vr
    (expected * probe.float()).sum().backward()
    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.03)
    for a, reference in zip(actual_grads, (qr, kr, vr, gr)):
        torch.testing.assert_close(a, reference.grad, atol=0.06, rtol=0.10)
    return {"length": length, "decay": decay}
