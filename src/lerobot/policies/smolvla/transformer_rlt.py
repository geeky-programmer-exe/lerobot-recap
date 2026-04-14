import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # batch first format (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]   ← now batch-first
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRLT(nn.Module):
    """
    Encoder-decoder transformer that learns a compact RL token (z_rl) from
    the VLM's final-layer token embeddings.

    Architecture (per the RLT paper):
      - ENCODER: takes [z1, ..., zM, e_rl] as input, where z_i are the VLM
        output embeddings for the M input tokens and e_rl is a LEARNED special
        token appended at the END. The encoder output at the <rl> position
        (last position) is z_rl — the information bottleneck.
      - DECODER: given only z_rl as memory, autoregressively reconstructs the
        original embeddings [z1, ..., zM] via teacher forcing during training.
        Because z_rl must retain enough information for full reconstruction,
        it is forced to be a compressed summary of the entire input.

    Changes from the original stub:
      - [FIXED] e_rl shape: was (1, input_classes), now (1, 1, d_model) so it
        is a proper sequence token that broadcasts over the batch dimension.
      - [FIXED] e_rl position: was prepended to the front; now APPENDED to the
        end per the paper ("the last input to the encoder being the RL token").
      - [FIXED] Encoder/decoder now use batch_first=True and operate on tensors
        of shape (batch, seq, d_model). The original nn.Transformer used
        seq-first format which was inconsistent with VLM outputs.
      - [FIXED] Decoder inputs: original code passed src as the query and z_rl
        as the memory, which is backwards. Now the decoder query is the
        teacher-forced shifted embeddings [BOS, z1, ..., z_{M-1}] and the
        memory is z_rl (the bottleneck), with a causal mask for autoregression.
      - [NEW] input_proj: projects from VLM hidden dimension (e.g. 2048) to
        the RLT's internal d_model before the transformer layers.
      - [NEW] decoder_bos: a learned beginning-of-sequence token fed as the
        first decoder input so the model can predict z1 with no prior context.
      - [CHANGED] output_proj: projects decoder outputs back to VLM embedding
        dimension so reconstruction loss is in the same space as the inputs.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_ff: int = 2048,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of the incoming VLM token embeddings (vlm_hidden_size).
            d_model:   Internal transformer dimension. May differ from input_dim.
            nhead:     Number of attention heads (must divide d_model evenly).
            num_encoder_layers: Depth of the RLT encoder.
            num_decoder_layers: Depth of the RLT decoder.
            dim_ff:    Feed-forward hidden dimension inside each transformer layer.
            dropout:   Dropout probability.
        """
        super(TransformerRLT, self).__init__()

        # [NEW] Project VLM embeddings from input_dim -> d_model before the transformer.
        # This allows the RLT to use a smaller internal dimension than the VLM.
        # Uses Identity when input_dim == d_model (no projection needed).
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # [FIXED] e_rl is the learned embedding for the <rl> special token.
        # Shape (1, 1, d_model) broadcasts over the batch dimension at runtime.
        self.e_rl = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding, now in batch-first format (batch, seq, d_model).
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # [CHANGED] Separate encoder and decoder with batch_first=True so tensors
        # have shape (batch, seq, d_model) throughout, matching VLM output format.
        # Original code used nn.Transformer which defaults to seq-first format and
        # did not expose encoder/decoder separately in a clean way.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # [NEW] Learned beginning-of-sequence token for the decoder.
        # At decoding step 0 there is no previous embedding, so BOS acts as
        # the dummy input that allows the model to predict z1.
        # Shape (1, 1, d_model) broadcasts over batch.
        self.decoder_bos = nn.Parameter(torch.randn(1, 1, d_model))

        # Output projection: decoder hidden states -> VLM embedding space.
        self.output_proj = nn.Linear(d_model, input_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor):
        """
        Forward pass: encode z into z_rl and decode back to reconstruct z.

        Args:
            z: VLM final-layer token embeddings, shape (batch, M, input_dim).
               These are produced by the frozen pretrained VLA for the current
               state/language input; denoted z_{1:M} in the paper.

        Returns:
            z_rl:    RL token, shape (batch, d_model). This is the bottleneck
                     embedding used downstream for RL value estimation.
            z_recon: Reconstructed embeddings, shape (batch, M, input_dim).
                     Used to compute the autoregressive reconstruction loss
                     that trains the RLT (and implicitly z_rl).
        """
        batch, M, _ = z.shape

        # --- 1. Project input embeddings to internal dimension ---
        z_proj = self.input_proj(z)  # (batch, M, d_model)

        # --- 2. Add positional encoding (batch-first: (batch, seq, d_model)) ---
        z_proj = self.pos_enc(z_proj)  # (batch, M, d_model)

        # --- 3. Append <rl> token to the END of the encoder sequence ---
        # [FIXED] Original code prepended e_rl to the front. Per the paper,
        # e_rl is the LAST input so that the last encoder output is z_rl.
        e_rl = self.e_rl.expand(batch, -1, -1)       # (batch, 1, d_model)
        enc_input = torch.cat([z_proj, e_rl], dim=1)  # (batch, M+1, d_model)

        # --- 4. Run encoder; extract z_rl from the last position ---
        enc_out = self.encoder(enc_input)   # (batch, M+1, d_model)
        # [FIXED] Original took enc_out[-1] which is the last *batch* element in
        # seq-first format. Now we index the last *sequence* position correctly.
        z_rl_mem = enc_out[:, -1:, :]      # (batch, 1, d_model) — kept for cross-attn
        z_rl = z_rl_mem.squeeze(1)         # (batch, d_model)    — return value

        # --- 5. Autoregressive decoder: reconstruct z from z_rl ---
        # Teacher-forcing: shift the target sequence right by one and prepend BOS.
        #   Decoder input  : [BOS, z_proj_0, ..., z_proj_{M-1}]   length M
        #   Decoder target : [z_proj_0,  ..., z_proj_{M-1}, z_M]  length M
        # At step i the decoder sees positions 0..i and must predict position i.
        bos = self.decoder_bos.expand(batch, -1, -1)             # (batch, 1, d_model)
        dec_input = torch.cat([bos, z_proj[:, :-1, :]], dim=1)   # (batch, M, d_model)

        # Causal mask: position i can only attend to positions 0..i.
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(M, device=z.device)

        # [FIXED] Original passed src as query and z_rl as memory, which reversed
        # the roles. The correct call: query=dec_input, memory=z_rl_mem (bottleneck).
        dec_out = self.decoder(
            tgt=dec_input,
            memory=z_rl_mem,
            tgt_mask=tgt_mask,
        )  # (batch, M, d_model)

        # --- 6. Project decoder output back to VLM embedding dimension ---
        z_recon = self.output_proj(dec_out)  # (batch, M, input_dim)

        return z_rl, z_recon

    # ------------------------------------------------------------------
    # Inference helper: encode only (no decoding needed for RL)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Extract z_rl from z without running the decoder.
        Use this at inference time when you only need the RL token.

        Args:
            z: VLM embeddings, shape (batch, M, input_dim).

        Returns:
            z_rl: shape (batch, d_model).
        """
        batch, _, _ = z.shape
        z_proj = self.input_proj(z)
        z_proj = self.pos_enc(z_proj)
        e_rl = self.e_rl.expand(batch, -1, -1)
        enc_input = torch.cat([z_proj, e_rl], dim=1)
        enc_out = self.encoder(enc_input)
        return enc_out[:, -1, :]  # (batch, d_model)
