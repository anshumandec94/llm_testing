"""
SASRec, ported to this codebase with a second head that predicts ratings.

Reference:
  Wang-Cheng Kang and Julian McAuley, "Self-Attentive Sequential
  Recommendation", ICDM 2018.
  TensorFlow original: https://github.com/kang205/SASRec
  PyTorch port:        https://github.com/pmixer/SASRec.pytorch

**pmixer is authoritative wherever the two references disagree**, because the
port target is PyTorch. Every divergence found is listed below rather than
silently resolved. This module is a derived work of both; see ``NOTICES``.

The backbone is the reference architecture. What is new here is the second
head. The canonical SASRec objective is ranking: BCE over one sampled negative
per position, scored as a dot product against the item embedding table. That
produces a score with no defined unit, which is the exact problem this
benchmark exists to avoid. So a rating head predicts the target's **debiased
residual** directly, in debiased-rating units, and that is the measurement
value. The two are trained jointly as ``L = L_bce + rating_loss_weight * L_mse``
over one shared backbone.

Calibrating the ranking score post hoc was considered and rejected: a monotone
map is rank-preserving, so it cannot reorder anything and does nothing but set
units, which makes MAE tunable by changing the map's capacity rather than the
model.

Rating context can also be injected into the *input* sequence, behind
``sasrec_inject_rating``. Position t then carries
``item_emb[i] + (residual / residual_std) * w`` for a single learned direction
``w``. Only input positions may carry it. The target's residual is the label,
and injecting it at target positions leaks the answer; #18 owns the regression
test for that.

Divergences between kang205 (TF) and pmixer (PyTorch)
-----------------------------------------------------

1. **Block normalisation order.** kang205 normalises before each sublayer:
   ``multihead_attention(queries=normalize(seq), keys=seq)`` then
   ``feedforward(normalize(seq))``. pmixer's default is post-norm,
   ``LN(seq + sublayer(seq))``, with a ``--norm_first`` flag for the other
   order. We default to pmixer's post-norm via ``sasrec_norm_first=False``.
   Note that pmixer's pre-norm branch is *still* not kang205: pmixer feeds
   ``LN(x)`` as all of Q, K and V, while kang205 uses ``Q = LN(x)`` with
   ``K = V = x``.

2. **What the residual adds.** kang205 adds the *normalised* input back
   (``outputs += queries``, where ``queries`` is ``normalize(seq)``), so a
   block computes ``sublayer(LN(x)) + LN(x)`` rather than the usual
   ``sublayer(LN(x)) + x``. pmixer's pre-norm branch adds the raw ``x``. We
   follow pmixer.

3. **Padded timesteps as attention keys.** kang205 masks them out inside its
   attention (``key_masks`` from ``sign(reduce_sum(abs(keys)))``, set to
   -2**32+1 before the softmax). pmixer calls ``MultiheadAttention`` without a
   ``key_padding_mask``, so a padded position is a real, attendable key: its
   projected value is the bias term, which is constant but not zero, and it
   dilutes the attention distribution. Since sequences are left-padded and the
   mask is causal, every position can see every pad. We default to pmixer's
   behaviour and expose ``sasrec_mask_padded_keys`` to switch to kang205's,
   because this is a genuine modelling difference and guessing is worse than
   measuring.

4. **Zeroing padded timesteps.** kang205 applies ``seq *= mask`` after the
   embedding dropout and after each block. pmixer did the same via a
   ``timeline_mask`` in earlier revisions but its current ``main`` has dropped
   it entirely. We keep the zeroing: it is on this epic's parity checklist, it
   matches the published original, and without it LayerNorm gives padded
   positions non-zero state that then flows into the pooled representation.

5. **Positional embedding indexing.** kang205 uses a table of size ``maxlen``
   indexed from 0 with no padding row. pmixer uses ``maxlen + 1`` with
   ``padding_idx=0`` and indexes from 1, zeroing the index at padded positions
   so they get the zero vector. We follow pmixer.

6. **Feed-forward dropout placement.** kang205 is conv -> relu -> dropout ->
   conv -> dropout. pmixer is conv -> dropout -> relu -> conv -> dropout, so
   the inner dropout lands before the activation rather than after. We follow
   pmixer, but the two are **exactly equal**, not merely close: dropout
   multiplies by a non-negative scalar (0 or 1/(1-p)) and ReLU is positively
   homogeneous, so ``relu(dropout(x)) == dropout(relu(x))`` elementwise for any
   mask. This one is a difference in source only, and no experiment can
   distinguish the two. ``tests/test_sasrec_model.py`` pins the commutation.
   Both also use kernel-size-1 1-D convolutions rather than dense layers, which
   are equivalent in arithmetic and differ only in weight layout.

7. **L2 on embeddings.** kang205 regularises both the item and positional
   tables through TensorFlow's regulariser collection. pmixer applies
   ``l2_emb`` to the item table only. Both default the coefficient to 0.0. This
   is a training-loop concern, so it belongs to #18, not here.

8. **Loss reduction.** kang205 sums the positive and negative terms over
   non-padded positions and divides once by their count. pmixer takes two
   separately-averaged ``BCEWithLogitsLoss`` calls over the same index set and
   adds them. Over one index set these are equal; we follow pmixer's form.

9. **No output projection in the TensorFlow attention.** kang205 projects Q, K
   and V, splits the heads, concatenates them back and adds the residual, with
   no ``W_O``. ``torch.nn.MultiheadAttention``, which pmixer and this port use,
   applies ``out_proj``, an extra ``hidden x hidden`` matrix per block. This is
   the largest structural difference between the two references. It also
   interacts with ``num_heads``: without ``W_O`` kang205's heads never mix, so
   at ``num_heads > 1`` the two are different models rather than differently
   parameterised ones. We follow pmixer, which is why ``num_heads`` defaults to
   the reference's 1.

10. **Query masking.** kang205 zeroes the softmax weights of padded *query*
    rows before the weighted sum, so a padded query's attention output is
    exactly zero before its residual add. pmixer has no equivalent, and neither
    do we. This is distinct from divergence 3, which is about padded *keys*,
    and from divergence 4, which zeroes after the residual and the LayerNorm
    rather than inside the attention. We re-introduced kang205's post-block
    zeroing because the parity checklist asks for it, and it makes the query
    mask redundant for anything downstream: a padded position's state is forced
    to zero at the end of every block either way. The two differ only in what
    the padded position contributes *within* a block, which nothing reads,
    since padded targets are masked out of both losses.

Not a divergence, but worth stating: both scale the item embeddings by
``sqrt(hidden_units)`` before the first block and neither scales the positional
embeddings.

One deliberate cosmetic departure from pmixer: it transposes to
``(seq, batch, hidden)`` around every attention call because it predates
``batch_first``. We pass ``batch_first=True`` instead. The arithmetic is
identical and the tensors stay in one layout throughout.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from sim.agents.sasrec_data import PAD_INDEX


class PointWiseFeedForward(nn.Module):
    """
    Position-wise feed-forward block, following pmixer exactly.

    Kernel-size-1 1-D convolutions rather than dense layers, matching both
    references. The inner dropout sits before the ReLU, which is pmixer's
    ordering and divergence 6 from kang205.
    """

    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Conv1d wants (N, C, L), the sequence carries (N, L, C).
        outputs = self.conv1(inputs.transpose(-1, -2))
        outputs = self.conv2(self.relu(self.dropout1(outputs)))
        return self.dropout2(outputs).transpose(-1, -2)


@dataclass
class SasrecLosses:
    """The two objectives and their weighted total."""

    bce: torch.Tensor
    mse: torch.Tensor
    total: torch.Tensor


class SASRec(nn.Module):
    """
    The reference SASRec backbone with a ranking head and a rating head.

    ``item_num`` is the number of real items; the embedding table is one row
    larger because index 0 is reserved for padding, matching
    ``sim.agents.sasrec_data``.
    """

    def __init__(
        self,
        item_num: int,
        hidden_units: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout_rate: float = 0.2,
        maxlen: int = 200,
        norm_first: bool = False,
        mask_padded_keys: bool = False,
        inject_rating: bool = True,
        rating_head_hidden: int = 64,
        rating_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.item_num = item_num
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.norm_first = norm_first
        self.mask_padded_keys = mask_padded_keys
        self.inject_rating = inject_rating
        self.rating_loss_weight = rating_loss_weight

        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=PAD_INDEX)
        # maxlen + 1 rows indexed from 1, so padded positions can take index 0
        # and receive the zero vector. Divergence 5.
        self.pos_emb = nn.Embedding(maxlen + 1, hidden_units, padding_idx=PAD_INDEX)
        self.emb_dropout = nn.Dropout(p=dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    hidden_units, num_heads, dropout=dropout_rate, batch_first=True
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.forward_layers.append(
                PointWiseFeedForward(hidden_units, dropout_rate)
            )

        # The single learned direction the normalised residual is projected
        # onto when rating context is injected.
        self.rating_direction = nn.Parameter(torch.zeros(hidden_units))

        # Rating head: sequence state at t concatenated with the target item's
        # embedding, mapped to one scalar in debiased-rating units. Held as two
        # named layers rather than an nn.Sequential that also aliases them,
        # which would put four keys in state_dict for two tensors.
        self.rating_hidden = nn.Linear(2 * hidden_units, rating_head_hidden)
        self.rating_activation = nn.ReLU()
        self.rating_output = nn.Linear(rating_head_hidden, 1)

        self.init_weights()

    def init_weights(self) -> None:
        """
        Xavier-normal initialisation, following pmixer's training script.

        This is part of the reference model rather than incidental training
        glue, and leaving it to torch's defaults is not neutral: ``nn.Embedding``
        defaults to N(0, 1), which ``embed_inputs`` then multiplies by
        ``sqrt(hidden_units)``, so first-block activations come out roughly an
        order of magnitude larger than the reference. Nothing would fail; the
        model would just train badly, and a backend comparison would quietly
        be unfair to SASRec. So it lives here, where every caller gets it, not
        in the training script where it can be forgotten.

        Parameters with fewer than two dimensions have no fan-in/fan-out and
        are left at their defaults, matching pmixer's guarded loop. The padding
        rows are re-zeroed afterwards because the initialiser overwrites them.
        """
        for param in self.parameters():
            if param.dim() >= 2:
                nn.init.xavier_normal_(param)

        nn.init.normal_(self.rating_direction, std=0.02)

        with torch.no_grad():
            self.item_emb.weight[PAD_INDEX].fill_(0.0)
            self.pos_emb.weight[PAD_INDEX].fill_(0.0)

    @classmethod
    def from_config(cls, config, item_num: int) -> "SASRec":
        """Build from the ``sasrec_*`` fields of a :class:`~sim.config.SimConfig`."""
        return cls(
            item_num=item_num,
            hidden_units=config.sasrec_hidden_units,
            num_blocks=config.sasrec_num_blocks,
            num_heads=config.sasrec_num_heads,
            dropout_rate=config.sasrec_dropout_rate,
            maxlen=config.sasrec_maxlen,
            norm_first=config.sasrec_norm_first,
            mask_padded_keys=config.sasrec_mask_padded_keys,
            inject_rating=config.sasrec_inject_rating,
            rating_head_hidden=config.sasrec_rating_head_hidden,
            rating_loss_weight=config.sasrec_rating_loss_weight,
        )

    @property
    def device(self) -> torch.device:
        return self.item_emb.weight.device

    # ──────────────────────────────────────────────────────────────────────
    # Backbone
    # ──────────────────────────────────────────────────────────────────────

    def embed_inputs(
        self,
        log_seqs: torch.Tensor,
        residuals: torch.Tensor | None = None,
        residual_std: float = 1.0,
    ) -> torch.Tensor:
        """
        Input embeddings before the first block: item, position and, optionally,
        rating context.

        Exposed separately so the injection flag can be tested by comparing
        tensors rather than by reading loss curves.
        """
        if log_seqs.shape[1] > self.maxlen:
            raise ValueError(
                f"sequence length {log_seqs.shape[1]} exceeds maxlen "
                f"{self.maxlen}; the positional table has no row for it"
            )
        seqs = self.item_emb(log_seqs) * (self.hidden_units**0.5)

        # Positions are indexed from 1, leaving row 0 free as the padding row.
        positions = torch.arange(
            1, log_seqs.shape[1] + 1, device=log_seqs.device
        ).unsqueeze(0).expand(log_seqs.shape[0], -1)
        # Padded positions take index 0 and so get the zero vector.
        positions = positions * (log_seqs != PAD_INDEX)
        seqs = seqs + self.pos_emb(positions)

        if self.inject_rating and residuals is not None:
            if residual_std <= 0.0:
                raise ValueError(
                    f"residual_std must be positive to normalise the injected "
                    f"rating signal, got {residual_std}"
                )
            scaled = (residuals / residual_std).unsqueeze(-1)
            seqs = seqs + scaled * self.rating_direction

        return seqs

    def log2feats(
        self,
        log_seqs: torch.Tensor,
        residuals: torch.Tensor | None = None,
        residual_std: float = 1.0,
    ) -> torch.Tensor:
        """
        Run the backbone. ``(batch, maxlen)`` in, ``(batch, maxlen, hidden)`` out.

        Position t of the output summarises positions 0..t of the input and
        nothing later, which is what the causal mask enforces.
        """
        seqs = self.embed_inputs(log_seqs, residuals, residual_std)
        seqs = self.emb_dropout(seqs)

        # Divergence 4: kang205 zeroes padded timesteps here and after every
        # block; pmixer's current main does not. We keep the zeroing.
        padding_mask = log_seqs == PAD_INDEX
        seqs = seqs * ~padding_mask.unsqueeze(-1)

        seq_len = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=seqs.device)
        )
        # Divergence 3: off by default, matching pmixer. A fully padded row
        # masks every key, which torch 2.10 resolves to zeros rather than the
        # NaN an unguarded softmax over -inf would give; tested, not assumed.
        key_padding_mask = padding_mask if self.mask_padded_keys else None

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                normed = self.attention_layernorms[i](seqs)
                attn_out, _ = self.attention_layers[i](
                    normed,
                    normed,
                    normed,
                    attn_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                seqs = seqs + attn_out
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                attn_out, _ = self.attention_layers[i](
                    seqs,
                    seqs,
                    seqs,
                    attn_mask=attention_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=False,
                )
                seqs = self.attention_layernorms[i](seqs + attn_out)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

            seqs = seqs * ~padding_mask.unsqueeze(-1)

        return self.last_layernorm(seqs)

    # ──────────────────────────────────────────────────────────────────────
    # Heads
    # ──────────────────────────────────────────────────────────────────────

    def rank_logits(
        self, log_feats: torch.Tensor, item_seqs: torch.Tensor
    ) -> torch.Tensor:
        """Ranking score: dot product of the sequence state against item embeddings."""
        return (log_feats * self.item_emb(item_seqs)).sum(dim=-1)

    def predict_residual(
        self, log_feats: torch.Tensor, item_seqs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicted debiased residual for each target, in debiased-rating units.

        This is the benchmark's measurement value. Add the item's bias baseline
        back to recover a rating.
        """
        paired = torch.cat([log_feats, self.item_emb(item_seqs)], dim=-1)
        return self._rating_mlp(paired).squeeze(-1)

    def _rating_mlp(self, paired: torch.Tensor) -> torch.Tensor:
        return self.rating_output(self.rating_activation(self.rating_hidden(paired)))

    def forward(
        self,
        log_seqs: torch.Tensor,
        pos_seqs: torch.Tensor,
        neg_seqs: torch.Tensor,
        input_residuals: torch.Tensor | None = None,
        residual_std: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Returns ``(pos_logits, neg_logits, predicted_residuals)``, each
        ``(batch, maxlen)``. ``input_residuals`` describes the *input*
        positions only; the target residuals are labels and must never be fed
        in here.
        """
        log_feats = self.log2feats(log_seqs, input_residuals, residual_std)
        return (
            self.rank_logits(log_feats, pos_seqs),
            self.rank_logits(log_feats, neg_seqs),
            self.predict_residual(log_feats, pos_seqs),
        )

    def losses(
        self,
        pos_logits: torch.Tensor,
        neg_logits: torch.Tensor,
        predicted_residuals: torch.Tensor,
        target_residuals: torch.Tensor,
        pos_seqs: torch.Tensor,
    ) -> SasrecLosses:
        """
        The joint objective, masked to non-padded target positions.

        A padded target contributes exactly zero to both terms. Following
        pmixer, the positive and negative BCE terms are averaged separately and
        added, which equals kang205's single sum over the same index set.
        """
        valid = pos_seqs != PAD_INDEX
        if not bool(valid.any()):
            # Stay attached to the graph so a backward pass still works, but
            # do not route a possible NaN logit through the multiplication.
            zero = (pos_logits * 0.0).nan_to_num().sum()
            return SasrecLosses(bce=zero, mse=zero, total=zero)

        bce_fn = nn.functional.binary_cross_entropy_with_logits
        pos_sel = pos_logits[valid]
        neg_sel = neg_logits[valid]
        bce = bce_fn(pos_sel, torch.ones_like(pos_sel)) + bce_fn(
            neg_sel, torch.zeros_like(neg_sel)
        )

        mse = nn.functional.mse_loss(
            predicted_residuals[valid], target_residuals[valid]
        )
        return SasrecLosses(
            bce=bce, mse=mse, total=bce + self.rating_loss_weight * mse
        )

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_items(
        self,
        log_seqs: torch.Tensor,
        item_indices: torch.Tensor,
        input_residuals: torch.Tensor | None = None,
        residual_std: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score candidate items from the final sequence position.

        Returns ``(rank_logits, predicted_residuals)``, both
        ``(batch, n_candidates)``. The residual is what the benchmark reports;
        the logit is the canonical SASRec score and is kept for ranking
        diagnostics.

        Sequences are left-padded, so the final position is always the most
        recent interaction and is never padding for a user with any history.

        Dropout is disabled for the duration regardless of the module's mode.
        This reads as a self-contained inference entry point and will be used
        as one, and scoring a model left in ``train()`` would otherwise apply
        dropout and return a different answer on every call.
        """
        was_training = self.training
        self.eval()
        try:
            log_feats = self.log2feats(log_seqs, input_residuals, residual_std)
            final = log_feats[:, -1, :]

            item_embs = self.item_emb(item_indices)
            logits = torch.einsum("bch,bh->bc", item_embs, final)

            expanded = final.unsqueeze(1).expand(-1, item_indices.shape[1], -1)
            residuals = self._rating_mlp(
                torch.cat([expanded, item_embs], dim=-1)
            ).squeeze(-1)
        finally:
            if was_training:
                self.train()
        return logits, residuals
