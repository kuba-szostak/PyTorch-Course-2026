"""
Part 2 (Lab 9): MultiHeadAttention and Block.
Task: MultiHeadAttention (multiple AttentionHeads, concat, project), Block (LayerNorm -> MHA/FFN -> residual).
"""
import torch
import torch.nn as nn

from part_1 import FeedForward, AttentionHead, _test_config


class MultiHeadAttention(nn.Module):
    """Multiple AttentionHead modules in parallel; concat outputs then project back to n_embd."""
    def __init__(self, config, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(config, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """Transformer block: LayerNorm -> MultiHeadAttention -> residual; LayerNorm -> FeedForward -> residual."""

    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = MultiHeadAttention(config, config.n_head, head_size)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ------ Tests ------

def test_mha_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run MultiHeadAttention, then output shape is (B,T,C) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    head_size = config.n_embd // config.n_head
    mha = MultiHeadAttention(config, config.n_head, head_size)
    out = mha(x)
    assert out.shape == (B, T, C), "MultiHeadAttention shape"
    assert not torch.isnan(out).any(), "MultiHeadAttention output should not be NaN"
    print("  test_mha_shape_and_finite [OK]")


def test_block_shape_and_finite():
    """Given a small config and (B,T,C) input, when we run Block, then output shape is (B,T,C) and has no NaN."""
    config = _test_config()
    B, T, C = 2, 4, config.n_embd
    x = torch.randn(B, T, C)
    block = Block(config)
    out = block(x)
    assert out.shape == (B, T, C), "Block shape"
    assert not torch.isnan(out).any(), "Block output should not be NaN"
    print("  test_block_shape_and_finite [OK]")


def main():
    print("=== Part 2: MultiHeadAttention and Block ===\n")
    test_mha_shape_and_finite()
    test_block_shape_and_finite()
    print("\nAll tests passed [OK]")


if __name__ == "__main__":
    main()
