import pytest
import torch


@pytest.fixture(autouse=True)
def exact_float32():
    """Importing ``scripts/train.py`` sets ``float32_matmul_precision`` to ``medium`` for the whole session, and
    CPUs with AMX / AVX512-BF16 then run float32 matmuls in bf16: parity tests fail on those hosts only."""
    torch.set_float32_matmul_precision("highest")
