import torch

from transformer_rnn import working_memory_transformer as mod


def test_WorkingMemoryTransformer():
    cls = mod.WorkingMemoryTransformer

    N, S = 8, 4
    W, P = 16, 2
    E, head = 4, 1
    wmt = cls(E, head, W, 1, E)
    assert wmt.working_memory_length == W
    assert wmt.push_length == 1
    assert wmt.use_pos_enc is True

    wmt = cls(E, head, W, 1, E, push_length=P, use_pos_enc=False)
    assert wmt.working_memory_length == W
    assert wmt.push_length == P
    assert wmt.use_pos_enc is False

    src = torch.randn(S, N, E)
    out_wm = wmt(src)
    assert torch.all(out_wm[:-P] == 0.0).bool()
    assert torch.any(out_wm[-P:] != 0.0).bool()

    sm = torch.ones(W, S, dtype=torch.bool)
    wmm = torch.ones(W, W, dtype=torch.bool)
    skpm = torch.ones(N, S, dtype=torch.bool)
    wmkpm = torch.ones(N, W, dtype=torch.bool)
    out_wm = wmt(src, out_wm, sm, wmm, skpm, wmkpm)
    assert torch.all(out_wm[: -P * 2] == 0.0).bool()
    assert torch.any(out_wm[-P * 2 :] != 0.0).bool()
