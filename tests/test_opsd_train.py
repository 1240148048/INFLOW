from __future__ import annotations

import pytest
import torch

from rig.opsd_train import reverse_kl


def test_chunked_reverse_kl_matches_unchunked_value_and_gradient() -> None:
    torch.manual_seed(7)
    teacher = torch.randn(2, 7, 19)
    student_full = torch.randn(2, 7, 19, requires_grad=True)
    student_chunked = student_full.detach().clone().requires_grad_(True)

    full = reverse_kl(student_full, teacher)
    chunked = reverse_kl(student_chunked, teacher, chunk_tokens=3)
    full.backward()
    chunked.backward()

    torch.testing.assert_close(chunked, full)
    torch.testing.assert_close(student_chunked.grad, student_full.grad)


def test_reverse_kl_rejects_nonpositive_chunk_size() -> None:
    logits = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="kl_chunk_tokens"):
        reverse_kl(logits, logits, chunk_tokens=0)
