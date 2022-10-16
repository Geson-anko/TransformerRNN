import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):

    pe: Tensor

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        factory_kwds = {"device": device, "dtype": dtype}

        position = torch.arange(max_len, **factory_kwds).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, **factory_kwds) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, **factory_kwds)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class WorkingMemoryTransformer(nn.Module):
    """Working Memory Transformer class.
    This is a transformer encoder (precisely 'decoder') that can encode input sequence
    together with `working_memory` and store that information as next `working_memory`.

    Created for expanding transformer model to RNN. The forward pass of this layer is,
    1step of RNN(TransformerRNN). `working_memory`  is precisely the `hidden_state` of the RNN.

    Note: This layer outputs next `working_memory` tensor only.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        working_memory_length: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        push_length: int = 1,
        use_pos_enc: bool = True,
        device=None,
        dtype=None,
        **transformer_decoder_kwds
    ) -> None:
        r"""Construct this layer. Most arguments are same with `TransformerDecoderLayer`.

        Args:
            - working_memory_length (int): If working_memory is not provided in the forward pass,
                generate it with this length.
            - num_layers (int): The number of transformer decoder layers.
            - push_length (int): Length of encoded working memory's to be added after input `working_memory`.
                If `push_length` is same with `working_memory_length`, input `working_memory` is completely
                replaced by a new working memory (transformer decoder output).
            - use_pos_enc (bool): If True, apply positional encoding to input `working_memory`.
            - **transformer_decoder_kwds (dict): Other arguments of `TransformerDecoderLayer` class.

        Note: `batch_first` is not supported.
        """
        super().__init__()

        factory_kwds = {"device": device, "dtype": dtype}

        self.working_memory_length = working_memory_length
        self.push_length = push_length
        self.use_pos_enc = use_pos_enc

        if use_pos_enc:
            self.pos_encoder = PositionalEncoding(
                d_model, dropout, max_len=working_memory_length, **factory_kwds
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            **transformer_decoder_kwds,
            **factory_kwds,
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(
        self,
        src: Tensor,
        working_memory: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        working_memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        working_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the transformer decoder layer.
        Note: :math:`tgt=working_memory` :math:`memory=src` in the `TransformerDecoder`.
        Note: Only supporting batched Tensor. Ex: :math:`(S, N, E)`

        Args:
            src: The sequence of encoding infomation. The sequence from the last layer of
                the encoder is often assumed.
            working_memory: Defaults to zeros if not provided.
            src_mask: The mask for the src sequence.
            working_memory_mask: The mask for the working memory.
            src_key_padding_mask: The mask for the src keys per batch.
            working_memory_key_padding_mask: The mask for the working memory keys per batch.

        Returns:
            next_wm: The working memory for next time step.

        Shape:
            - src: :math: `(S, N, E)`
            - working_memory: :math: `(W, N, E)`
            - src_mask: :math: `(W, S)`
            - working_memory_mask: :math: `(W, W)`
            - src_key_padding_mask: :math: `(N, S)`
            - working_memory_key_padding_mask: :math: `(N, W)`
            - next_wm: :math: `(W, N, E)`
        """

        if working_memory is None:
            working_memory = torch.zeros(
                (self.working_memory_length, src.size(1), src.size(2)),
                device=src.device,
                dtype=src.dtype,
            )

        if self.use_pos_enc:
            working_memory = self.pos_encoder(working_memory)

        wm_t1 = self.transformer_decoder(
            working_memory,
            src,
            tgt_mask=working_memory_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=working_memory_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        next_wm = torch.cat(
            [working_memory[self.push_length :], wm_t1[-self.push_length :]], dim=0  # type: ignore
        )

        return next_wm
