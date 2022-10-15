import torch
import torchinfo

from transformer_rnn import WorkingMemoryTransformer

batch_size = 4
src_length = 8
dim = 4
wm_length = 3
push_length = 1
use_pos_enc = True

wmt = WorkingMemoryTransformer(dim, 1, wm_length, 1, 512, 0.0, push_length, use_pos_enc)

src = torch.randn(src_length, batch_size, dim)
wm_out = wmt(src)
print(wm_out)
print(wm_out.shape)

torchinfo.summary(
    wmt, input_size=[(src_length, batch_size, dim), (wm_length, batch_size, dim)]
)

time_steps = 10
