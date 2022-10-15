from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .working_memory_transformer import WorkingMemoryTransformer


class TransformerRNN(nn.Module):
    r"""Transformer RNN class
    Expand Transformer Decoder to RNN.
    """

    def __init__(
        self,
        working_memory_transformer: WorkingMemoryTransformer,
        batch_first: bool = False,
    ) -> None:
        r"""Construct TransformerRNN.

        Args:
            - working_memory_transformer (WorkingMemoryTransformer): An instance of WorkingMemoryTransformer.
            - batch_first (bool): If ``True``, then the input and output tensor are provided
                as (batch, time, seq, feature). Default: ``False`` (time, seq, batch, feature).
        """
        super().__init__()

        self.working_memory_transformer = working_memory_transformer
        self.batch_first = batch_first

    def forward(
        self,
        src: Tensor,
        initial_working_memory: Optional[Tensor] = None,
        src_masks: Optional[Tensor] = None,
        working_memory_masks: Optional[Tensor] = None,
        src_key_padding_masks: Optional[Tensor] = None,
        working_memory_key_padding_masks: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the working memory transformer.
        Note: Only supporting batched Tensor.

        Args:
            src: The time sequence of encoding infomation sequence.
            intial_working_memory: Inital working memory for `WorkingMemoryTransformer`.
            src_masks: The mask or masks of time series for the src.
            working_memory_masks: The mask or masks of time series for the working_memory.
            src_key_padding_masks: The mask or masks of time series for the src keys per batch.
            working_memory_key_padding_masks: The mask or masks of time series for the working memory keys per batch.

        Returns:
            next_working_memories: Time series of next working memory.

        Shape:
            - src: :math: `(T, S, N, E)`. If batch_first, :math: `(N, T, S, E)`.
            - intial_working_memory: :math:`(W, N, E)`. If batch_first, :math: `(N, W, E)`
            - src_masks: :math: `(S, S)` or `(T, S, S)`
            - working_memory_mask: :math: `(W, W)` or `(T, W, W)`.
            - src_key_padding_masks: :math:  `(N, S)` or `(T, N, S)`
            - working_memory_key_padding_masks: :math: `(N, W)` or `(T, N, W)`
            - next_working_memories: `(T, W, N, E)`
        """

        wm: Tensor = initial_working_memory  # type: ignore

        if self.batch_first:
            src = src.permute((1, 2, 0, 3))
            wm = wm.permute(1, 0, 2)

        next_working_memories = []

        for t in range(src.size(0)):
            s = src[t]
            sm = get_tensor_at_time(src_masks, t, 3)
            wmm = get_tensor_at_time(working_memory_masks, t, 3)
            skpm = get_tensor_at_time(src_key_padding_masks, t, 3)
            wmkpm = get_tensor_at_time(working_memory_key_padding_masks, t, 3)

            wm = self.working_memory_transformer(s, wm, sm, wmm, skpm, wmkpm)
            next_working_memories.append(wm)

        wm = torch.stack(next_working_memories)

        if self.batch_first:
            wm = wm.permute((2, 0, 1, 3)).contiguous()

        return wm


def get_tensor_at_time(
    tensor: Optional[Tensor], time: int, dim_with_time: int
) -> Optional[Tensor]:
    """Get the tensor at the time."""
    if tensor is not None:
        if tensor.dim() == dim_with_time:
            return tensor[time]
        else:
            return tensor
    else:
        return None
