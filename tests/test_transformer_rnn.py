import torch

from transformer_rnn import transformer_rnn as mod
from transformer_rnn.working_memory_transformer import WorkingMemoryTransformer


def test_get_tensor_at_time():
    f = mod.get_tensor_at_time

    assert f(None, 0, 1) is None
    assert int(f(torch.arange(10), 2, 1)) == 2
    assert torch.all(f(torch.arange(10), 4, 2) == torch.arange(10))


def test_TransformerRNN():
    cls = mod.TransformerRNN

    T, S, N, E, W = 16, 8, 4, 2, 1
    wmt = WorkingMemoryTransformer(E, 1, W, 1, 4)

    src = torch.randn(T, S, N, E)
    src_bf = torch.randn(N, T, S, E)
    ini_wm = torch.randn(W, N, E)
    ini_wm_bf = torch.randn(N, W, E)

    trnn = cls(wmt)
    assert trnn.working_memory_transformer is wmt
    assert trnn.batch_first is False

    out_wm = trnn(src)
    assert tuple(out_wm.shape) == (T, W, N, E)
    out_wm = trnn(src, ini_wm)
    assert tuple(out_wm.shape) == (T, W, N, E)

    trnn = cls(wmt, True)
    out_wm = trnn(src_bf, ini_wm_bf)
    assert trnn.batch_first is True
    assert tuple(out_wm.shape) == (N, T, W, E)

    sm = torch.ones(W, S, dtype=torch.bool)
    sms = torch.ones(T, W, S, dtype=torch.bool)
    wmm = torch.ones(W, W, dtype=torch.bool)
    wmms = torch.ones(T, W, W, dtype=torch.bool)
    skpm = torch.ones(N, S, dtype=torch.bool)
    skpms = torch.ones(T, N, S, dtype=torch.bool)
    wmkpm = torch.ones(N, W, dtype=torch.bool)
    wmkpms = torch.ones(T, N, W, dtype=torch.bool)

    trnn = cls(wmt, False)
    out_wm = trnn(src, ini_wm, sm, wmm, skpm, wmkpm)
    assert tuple(out_wm.shape) == (T, W, N, E)
    out_wm = trnn(src, ini_wm, sms, wmms, skpms, wmkpms)
    assert tuple(out_wm.shape) == (T, W, N, E)
