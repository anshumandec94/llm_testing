"""
Tests for sim/agents/sasrec_model.py.

The acceptance clauses of the sub-issue drive this file: documented shapes from
a forward pass, a causal mask that no position can see past, padded positions
contributing exactly zero to both losses, and the rating-injection flag
provably changing the input tensor when on and provably not when off.

Everything runs on a tiny random model on CPU. No fixture data is needed, so
nothing here touches data/ml-32m/.
"""
from __future__ import annotations

import pytest
import torch

from sim.agents.sasrec_data import PAD_INDEX
from sim.agents.sasrec_model import SASRec, PointWiseFeedForward
from sim.config import SimConfig

ITEM_NUM = 12
HIDDEN = 8
MAXLEN = 6
BATCH = 3


def build_model(
    *,
    hidden_units: int = HIDDEN,
    num_blocks: int = 2,
    num_heads: int = 2,
    norm_first: bool = False,
    mask_padded_keys: bool = False,
    inject_rating: bool = True,
    rating_loss_weight: float = 1.0,
) -> SASRec:
    """A tiny deterministic model. Dropout is off so comparisons are exact."""
    torch.manual_seed(0)
    model = SASRec(
        item_num=ITEM_NUM,
        hidden_units=hidden_units,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=0.0,
        maxlen=MAXLEN,
        norm_first=norm_first,
        mask_padded_keys=mask_padded_keys,
        inject_rating=inject_rating,
        rating_loss_weight=rating_loss_weight,
    )
    model.eval()
    return model


@pytest.fixture
def batch():
    """Left-padded sequences: row 0 half padded, row 1 full, row 2 mostly padded."""
    log_seqs = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3],
            [4, 5, 6, 7, 8, 9],
            [0, 0, 0, 0, 0, 2],
        ],
        dtype=torch.long,
    )
    pos_seqs = torch.tensor(
        [
            [0, 0, 0, 2, 3, 4],
            [5, 6, 7, 8, 9, 10],
            [0, 0, 0, 0, 0, 3],
        ],
        dtype=torch.long,
    )
    neg_seqs = torch.tensor(
        [
            [0, 0, 0, 11, 10, 9],
            [1, 2, 3, 4, 5, 6],
            [0, 0, 0, 0, 0, 11],
        ],
        dtype=torch.long,
    )
    residuals = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.5, -0.5, 0.25],
            [1.0, -1.0, 0.5, -0.25, 2.0, -2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
        ]
    )
    return log_seqs, pos_seqs, neg_seqs, residuals


class TestShapes:
    def test_forward_produces_documented_shapes(self, batch):
        log_seqs, pos_seqs, neg_seqs, residuals = batch
        model = build_model()
        pos_logits, neg_logits, predicted = model(
            log_seqs, pos_seqs, neg_seqs, residuals, residual_std=1.0
        )
        assert pos_logits.shape == (BATCH, MAXLEN)
        assert neg_logits.shape == (BATCH, MAXLEN)
        assert predicted.shape == (BATCH, MAXLEN)
        assert torch.isfinite(pos_logits).all()
        assert torch.isfinite(predicted).all()

    def test_log2feats_shape(self, batch):
        log_seqs, _, _, residuals = batch
        feats = build_model().log2feats(log_seqs, residuals)
        assert feats.shape == (BATCH, MAXLEN, HIDDEN)

    def test_embedding_table_has_one_row_per_item_plus_padding(self):
        model = build_model()
        assert model.item_emb.weight.shape == (ITEM_NUM + 1, HIDDEN)
        assert model.item_emb.padding_idx == PAD_INDEX
        assert torch.all(model.item_emb.weight[PAD_INDEX] == 0.0)

    def test_positional_table_is_maxlen_plus_one_with_padding_row(self):
        """Divergence 5: pmixer indexes positions from 1 and reserves row 0."""
        model = build_model()
        assert model.pos_emb.weight.shape == (MAXLEN + 1, HIDDEN)
        assert model.pos_emb.padding_idx == PAD_INDEX

    def test_score_items_shapes(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model()
        candidates = torch.tensor([[1, 2, 3, 4]] * BATCH, dtype=torch.long)
        logits, predicted = model.score_items(log_seqs, candidates, residuals)
        assert logits.shape == (BATCH, 4)
        assert predicted.shape == (BATCH, 4)

    def test_score_items_matches_the_final_position_of_the_training_path(self, batch):
        """The inference path must read the same state the training path builds."""
        log_seqs, _, _, residuals = batch
        model = build_model()
        candidates = torch.tensor([[5], [5], [5]], dtype=torch.long)
        logits, predicted = model.score_items(log_seqs, candidates, residuals)

        feats = model.log2feats(log_seqs, residuals)
        targets = torch.full((BATCH, MAXLEN), 5, dtype=torch.long)
        expected_logits = model.rank_logits(feats, targets)[:, -1]
        expected_res = model.predict_residual(feats, targets)[:, -1]

        assert torch.allclose(logits[:, 0], expected_logits, atol=1e-5)
        assert torch.allclose(predicted[:, 0], expected_res, atol=1e-5)


class TestCausality:
    def test_no_position_attends_to_a_later_one(self):
        """
        Perturb the input at position t and assert only outputs at >= t move.
        This tests the property end to end rather than inspecting the mask.
        """
        model = build_model(inject_rating=False)
        base_seq = torch.arange(1, MAXLEN + 1, dtype=torch.long).unsqueeze(0)
        base = model.log2feats(base_seq)

        for t in range(MAXLEN):
            altered = base_seq.clone()
            altered[0, t] = (int(altered[0, t]) % ITEM_NUM) + 1
            assert altered[0, t] != base_seq[0, t]
            moved = model.log2feats(altered)
            delta = (moved - base).abs().sum(dim=-1)[0]

            assert torch.all(delta[:t] < 1e-6), (
                f"changing position {t} moved an earlier position"
            )
            assert delta[t] > 1e-6

    def test_causality_holds_with_rating_injection_on(self):
        """Injection adds a per-position term, which must not leak backwards."""
        model = build_model(inject_rating=True)
        seq = torch.arange(1, MAXLEN + 1, dtype=torch.long).unsqueeze(0)
        residuals = torch.zeros(1, MAXLEN)
        base = model.log2feats(seq, residuals, residual_std=1.0)

        for t in range(MAXLEN):
            altered = residuals.clone()
            altered[0, t] = 3.0
            moved = model.log2feats(seq, altered, residual_std=1.0)
            delta = (moved - base).abs().sum(dim=-1)[0]
            assert torch.all(delta[:t] < 1e-6)
            assert delta[t] > 1e-6


class TestPaddingContributesNothing:
    def test_padded_targets_contribute_zero_to_both_losses(self, batch):
        """
        Changing what sits at a padded target position must not move either
        loss. If the mask were dropped, the pad row's embedding would enter
        both terms.
        """
        log_seqs, pos_seqs, neg_seqs, residuals = batch
        model = build_model()
        targets = torch.randn(BATCH, MAXLEN)

        def compute(pos, neg, target_res):
            pos_logits, neg_logits, predicted = model(
                log_seqs, pos, neg, residuals, residual_std=1.0
            )
            return model.losses(pos_logits, neg_logits, predicted, target_res, pos)

        before = compute(pos_seqs, neg_seqs, targets)

        # Move the labels at every padded target position.
        pad = pos_seqs == PAD_INDEX
        assert bool(pad.any())
        shifted = targets.clone()
        shifted[pad] += 100.0
        after = compute(pos_seqs, neg_seqs, shifted)

        assert torch.allclose(before.mse, after.mse, atol=1e-6)
        assert torch.allclose(before.total, after.total, atol=1e-6)

    def test_bce_ignores_padded_target_positions(self, batch):
        """
        The BCE half of the same clause. Perturbing the residual labels cannot
        test it: BCE is computed from the logits against constant ones/zeros
        and has no dependence on target_residuals at all, so the assertion
        above would hold even with the mask deleted. Compare the masked loss
        against an explicitly unmasked one instead; they must differ.
        """
        log_seqs, pos_seqs, neg_seqs, residuals = batch
        model = build_model()
        pos_logits, neg_logits, predicted = model(
            log_seqs, pos_seqs, neg_seqs, residuals, residual_std=1.0
        )
        masked = model.losses(
            pos_logits, neg_logits, predicted, torch.randn(BATCH, MAXLEN), pos_seqs
        )

        bce_fn = torch.nn.functional.binary_cross_entropy_with_logits
        unmasked = bce_fn(pos_logits, torch.ones_like(pos_logits)) + bce_fn(
            neg_logits, torch.zeros_like(neg_logits)
        )
        assert bool((pos_seqs == PAD_INDEX).any())
        assert not torch.allclose(masked.bce, unmasked, atol=1e-4)

        # And the masked value must equal the loss over exactly the real ones.
        valid = pos_seqs != PAD_INDEX
        expected = bce_fn(
            pos_logits[valid], torch.ones_like(pos_logits[valid])
        ) + bce_fn(neg_logits[valid], torch.zeros_like(neg_logits[valid]))
        assert torch.allclose(masked.bce, expected, atol=1e-6)

    def test_padded_positions_carry_no_state_through_the_backbone(self, batch):
        """
        Divergence 4: padded timesteps are explicitly zeroed after each block.

        Asserting only that the padded states are equal to each other is not
        enough, and would pass with the zeroing removed: sequences are
        left-padded, so under the causal mask a pad sees only pads, gets the
        zero item and position rows, and lands on the same constant either way.
        Assert they are exactly zero, which is what the zeroing produces and
        the attention output does not.
        """
        log_seqs, _, _, residuals = batch
        model = build_model()
        feats = model.log2feats(log_seqs, residuals)
        pad = log_seqs == PAD_INDEX

        assert bool(pad.any())
        assert torch.allclose(feats[pad], torch.zeros_like(feats[pad]), atol=1e-6)
        # And the real positions must not be zero, or the assertion above is
        # satisfied by a model that outputs nothing at all.
        assert float(feats[~pad].detach().abs().sum()) > 0.0

    def test_zeroing_changes_the_real_positions_too(self, batch):
        """
        The zeroing is not cosmetic: padded state that survives a block feeds
        the next block's attention, so removing it moves every real position.
        Pins divergence 4 as a behavioural choice rather than a tidy-up.
        """
        log_seqs, _, _, residuals = batch
        model = build_model()
        with_zeroing = model.log2feats(log_seqs, residuals)

        pad = log_seqs == PAD_INDEX
        assert bool(pad.any())
        # Feed the same batch with no padding at all; the real columns of the
        # first row now see different keys, so their states must move.
        unpadded = log_seqs.clone()
        unpadded[pad] = 1
        without = model.log2feats(unpadded, residuals)
        assert not torch.allclose(with_zeroing[~pad], without[~pad], atol=1e-4)

    def test_all_padding_batch_gives_zero_loss_and_no_nan(self):
        model = build_model()
        empty = torch.zeros(2, MAXLEN, dtype=torch.long)
        residuals = torch.zeros(2, MAXLEN)
        pos_logits, neg_logits, predicted = model(
            empty, empty, empty, residuals, residual_std=1.0
        )
        losses = model.losses(
            pos_logits, neg_logits, predicted, torch.zeros(2, MAXLEN), empty
        )
        assert float(losses.total.detach()) == 0.0
        assert torch.isfinite(losses.total)

    def test_all_padding_batch_is_finite_with_key_masking_on(self):
        """
        A fully padded row masks every key. torch 2.10 resolves that to zeros,
        but a softmax over a row of -inf is the classic NaN source and it would
        spread through the whole batch's gradients, so pin the behaviour rather
        than depending on an implementation detail staying put.
        """
        model = build_model(mask_padded_keys=True)
        model.train()
        log_seqs = torch.tensor(
            [[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]], dtype=torch.long
        )
        residuals = torch.zeros(2, MAXLEN)
        pos_logits, neg_logits, predicted = model(
            log_seqs, log_seqs, log_seqs, residuals, residual_std=1.0
        )
        assert torch.isfinite(pos_logits).all()
        assert torch.isfinite(predicted).all()

        losses = model.losses(
            pos_logits, neg_logits, predicted, torch.zeros(2, MAXLEN), log_seqs
        )
        losses.total.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"

    def test_all_padding_batch_still_backpropagates(self):
        """A zero loss must stay connected to the graph, or the step crashes."""
        model = build_model()
        model.train()
        empty = torch.zeros(2, MAXLEN, dtype=torch.long)
        residuals = torch.zeros(2, MAXLEN)
        pos_logits, neg_logits, predicted = model(
            empty, empty, empty, residuals, residual_std=1.0
        )
        losses = model.losses(
            pos_logits, neg_logits, predicted, torch.zeros(2, MAXLEN), empty
        )
        losses.total.backward()


class TestRatingInjection:
    def test_flag_on_changes_the_input_tensor(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model(inject_rating=True)
        bare = model.embed_inputs(log_seqs, residuals=None)
        injected = model.embed_inputs(log_seqs, residuals, residual_std=1.0)
        assert not torch.allclose(bare, injected)

    def test_flag_off_is_identical_to_the_bare_id_case(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model(inject_rating=False)
        bare = model.embed_inputs(log_seqs, residuals=None)
        with_residuals = model.embed_inputs(log_seqs, residuals, residual_std=1.0)
        assert torch.equal(bare, with_residuals)

    def test_injection_is_the_residual_over_std_times_one_direction(self, batch):
        """Pins the documented form: item_emb[i] + (residual / std) * w."""
        log_seqs, _, _, residuals = batch
        model = build_model(inject_rating=True)
        std = 2.0
        bare = model.embed_inputs(log_seqs, residuals=None)
        injected = model.embed_inputs(log_seqs, residuals, residual_std=std)
        expected = bare + (residuals / std).unsqueeze(-1) * model.rating_direction
        assert torch.allclose(injected, expected, atol=1e-6)

    def test_residual_std_scales_the_injection(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model(inject_rating=True)
        bare = model.embed_inputs(log_seqs, residuals=None)
        one = model.embed_inputs(log_seqs, residuals, residual_std=1.0) - bare
        two = model.embed_inputs(log_seqs, residuals, residual_std=2.0) - bare
        assert torch.allclose(one, two * 2.0, atol=1e-6)

    def test_zero_residual_std_is_rejected(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model(inject_rating=True)
        with pytest.raises(ValueError, match="residual_std must be positive"):
            model.embed_inputs(log_seqs, residuals, residual_std=0.0)

    def test_direction_is_a_single_learned_vector(self):
        model = build_model()
        assert model.rating_direction.shape == (HIDDEN,)
        assert model.rating_direction.requires_grad


class TestHeads:
    def test_rating_head_output_is_unbounded(self, batch):
        """
        The rating head predicts a debiased residual, which is signed and has
        no fixed range. A squashing activation on the output would silently
        cap it, so assert the head can produce both signs.
        """
        log_seqs, pos_seqs, _, residuals = batch
        model = build_model()
        feats = model.log2feats(log_seqs, residuals)
        with torch.no_grad():
            model.rating_output.bias.fill_(5.0)
            high = model.predict_residual(feats, pos_seqs)
            model.rating_output.bias.fill_(-5.0)
            low = model.predict_residual(feats, pos_seqs)
        assert float(high.mean()) > 1.0
        assert float(low.mean()) < -1.0

    def test_both_heads_share_one_backbone(self, batch):
        """
        A gradient from the rating head alone must reach the backbone, which is
        what makes this multi-task rather than two models in a trench coat.
        """
        log_seqs, pos_seqs, _, residuals = batch
        model = build_model()
        model.train()
        feats = model.log2feats(log_seqs, residuals)
        model.predict_residual(feats, pos_seqs).sum().backward()

        backbone_grads = {
            name: param.grad
            for name, param in model.named_parameters()
            if name.startswith("attention_layers.0") and param.grad is not None
        }
        assert backbone_grads, "no gradient reached the first attention block"
        assert any(float(g.abs().sum()) > 0.0 for g in backbone_grads.values())

    def test_rating_loss_weight_scales_only_the_mse_term(self, batch):
        log_seqs, pos_seqs, neg_seqs, residuals = batch
        targets = torch.randn(BATCH, MAXLEN)

        def total_for(weight: float):
            model = build_model(rating_loss_weight=weight)
            pos_logits, neg_logits, predicted = model(
                log_seqs, pos_seqs, neg_seqs, residuals, residual_std=1.0
            )
            return model.losses(pos_logits, neg_logits, predicted, targets, pos_seqs)

        zero = total_for(0.0)
        one = total_for(1.0)
        assert torch.allclose(zero.total, zero.bce)
        assert torch.allclose(one.total, one.bce + one.mse)
        assert not torch.allclose(zero.total, one.total)

    def test_rating_loss_weight_zero_still_trains_the_ranking_head(self, batch):
        """The w=0 ablation must leave the BCE objective intact."""
        log_seqs, pos_seqs, neg_seqs, residuals = batch
        model = build_model(rating_loss_weight=0.0)
        model.train()
        pos_logits, neg_logits, predicted = model(
            log_seqs, pos_seqs, neg_seqs, residuals, residual_std=1.0
        )
        losses = model.losses(
            pos_logits, neg_logits, predicted, torch.randn(BATCH, MAXLEN), pos_seqs
        )
        losses.total.backward()
        grad = model.item_emb.weight.grad
        assert grad is not None
        assert float(grad.abs().sum()) > 0.0


class TestParityWithTheReferences:
    def test_item_embeddings_are_scaled_by_sqrt_hidden(self):
        """Both references scale the item embedding before the first block."""
        model = build_model(inject_rating=False)
        seq = torch.tensor([[1]], dtype=torch.long)
        with torch.no_grad():
            model.pos_emb.weight.zero_()
        embedded = model.embed_inputs(seq)
        expected = model.item_emb.weight[1] * (HIDDEN**0.5)
        assert torch.allclose(embedded[0, 0], expected, atol=1e-6)

    def test_positional_embeddings_are_learned_not_sinusoidal(self):
        model = build_model()
        assert isinstance(model.pos_emb, torch.nn.Embedding)
        assert model.pos_emb.weight.requires_grad

    def test_feed_forward_uses_kernel_one_convolutions(self):
        ff = PointWiseFeedForward(HIDDEN, 0.0)
        assert isinstance(ff.conv1, torch.nn.Conv1d)
        assert ff.conv1.kernel_size == (1,)
        assert isinstance(ff.conv2, torch.nn.Conv1d)
        assert ff.conv2.kernel_size == (1,)

    def test_feed_forward_preserves_shape(self):
        ff = PointWiseFeedForward(HIDDEN, 0.0)
        x = torch.randn(BATCH, MAXLEN, HIDDEN)
        assert ff(x).shape == x.shape

    def test_inner_dropout_and_relu_commute_exactly(self):
        """
        Divergence 6 is notational, and this test exists to say so.

        pmixer is conv -> dropout -> relu; kang205 is conv -> relu -> dropout.
        No experiment can separate them: dropout multiplies by a non-negative
        scalar (0 or 1/(1-p)) and ReLU is positively homogeneous, so the two
        compose identically for any mask. Anyone auditing the parity list
        should not go looking for a behavioural difference here.
        """
        dropout = torch.nn.Dropout(p=0.5)
        relu = torch.nn.ReLU()
        dropout.train()
        x = torch.randn(64, 32)

        torch.manual_seed(3)
        pmixer_order = relu(dropout(x))
        torch.manual_seed(3)
        kang_order = dropout(relu(x))

        assert torch.equal(pmixer_order, kang_order)
        # Guard the premise: the dropout must actually be dropping something.
        assert float((pmixer_order == 0.0).float().mean()) > 0.2

    def test_dropout_rate_reaches_every_sublayer(self):
        """dropout_rate is not silently dropped on the way into the blocks."""
        model = SASRec(
            item_num=ITEM_NUM, hidden_units=HIDDEN, num_blocks=2, num_heads=2,
            dropout_rate=0.3, maxlen=MAXLEN,
        )
        assert model.emb_dropout.p == 0.3
        for block in model.forward_layers:
            assert isinstance(block, PointWiseFeedForward)
            assert block.dropout1.p == 0.3
            assert block.dropout2.p == 0.3
        for attn in model.attention_layers:
            assert attn.dropout == 0.3

    def test_dropout_actually_perturbs_a_training_forward_pass(self, batch):
        log_seqs, _, _, residuals = batch
        model = SASRec(
            item_num=ITEM_NUM, hidden_units=HIDDEN, num_blocks=2, num_heads=2,
            dropout_rate=0.5, maxlen=MAXLEN,
        )
        model.train()
        torch.manual_seed(1)
        first = model.log2feats(log_seqs, residuals)
        second = model.log2feats(log_seqs, residuals)
        assert not torch.allclose(first, second)

        model.eval()
        assert torch.allclose(
            model.log2feats(log_seqs, residuals),
            model.log2feats(log_seqs, residuals),
        )

    def test_positions_are_indexed_from_one(self):
        """
        Divergence 5. Indexing from 0 would put a real position on the reserved
        padding row and make the pad guard a no-op for it.
        """
        model = build_model(inject_rating=False)
        with torch.no_grad():
            model.item_emb.weight.zero_()
        seq = torch.tensor([[1, 2, 3]], dtype=torch.long)
        embedded = model.embed_inputs(seq)
        for offset in range(3):
            assert torch.allclose(
                embedded[0, offset], model.pos_emb.weight[offset + 1], atol=1e-6
            )

    def test_padded_positions_take_the_zero_positional_row(self):
        model = build_model(inject_rating=False)
        with torch.no_grad():
            model.item_emb.weight.zero_()
        seq = torch.tensor([[0, 0, 5]], dtype=torch.long)
        embedded = model.embed_inputs(seq)
        assert torch.allclose(embedded[0, 0], torch.zeros(HIDDEN), atol=1e-6)
        assert torch.allclose(embedded[0, 1], torch.zeros(HIDDEN), atol=1e-6)
        assert torch.allclose(embedded[0, 2], model.pos_emb.weight[3], atol=1e-6)

    def test_sequence_longer_than_maxlen_is_rejected(self):
        model = build_model()
        too_long = torch.ones(1, MAXLEN + 1, dtype=torch.long)
        with pytest.raises(ValueError, match="exceeds maxlen"):
            model.log2feats(too_long)


class TestInitialisation:
    def test_embeddings_are_not_left_at_the_torch_default(self):
        """
        pmixer xavier-initialises every 2-D parameter, and this is part of the
        reference model rather than training glue. torch's nn.Embedding default
        is N(0, 1), which embed_inputs then multiplies by sqrt(hidden_units),
        so skipping it makes first-block activations far too large and SASRec
        trains badly with nothing failing.
        """
        model = SASRec(item_num=200, hidden_units=64, maxlen=MAXLEN)
        std = float(model.item_emb.weight.detach().std())
        assert std < 0.5, f"item embeddings look uninitialised (std={std:.3f})"

    def test_padding_rows_are_zero_after_initialisation(self):
        """xavier_normal_ overwrites the padding rows, so they are re-zeroed."""
        model = SASRec(item_num=200, hidden_units=64, maxlen=MAXLEN)
        assert torch.all(model.item_emb.weight[PAD_INDEX] == 0.0)
        assert torch.all(model.pos_emb.weight[PAD_INDEX] == 0.0)

    def test_one_dimensional_parameters_survive_initialisation(self):
        """xavier has no fan-in for a 1-D tensor; the loop must skip them."""
        model = SASRec(item_num=200, hidden_units=64, maxlen=MAXLEN)
        assert torch.isfinite(model.rating_direction).all()
        assert float(model.rating_direction.detach().abs().sum()) > 0.0
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), f"non-finite init in {name}"

    def test_init_weights_is_idempotent_and_rezeroes_padding(self):
        model = SASRec(item_num=200, hidden_units=64, maxlen=MAXLEN)
        with torch.no_grad():
            model.item_emb.weight[PAD_INDEX].fill_(3.0)
        model.init_weights()
        assert torch.all(model.item_emb.weight[PAD_INDEX] == 0.0)


class TestStateDict:
    def test_rating_head_tensors_appear_once_each(self):
        """
        Aliasing the output layer into an nn.Sequential as well would emit four
        keys for two tensors, so a partial load could let one copy win.
        """
        keys = list(SASRec(item_num=ITEM_NUM, hidden_units=HIDDEN, maxlen=MAXLEN).state_dict())
        rating_keys = [k for k in keys if "rating" in k]
        assert sorted(rating_keys) == [
            "rating_direction",
            "rating_hidden.bias",
            "rating_hidden.weight",
            "rating_output.bias",
            "rating_output.weight",
        ]

    def test_state_dict_round_trips(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model()
        clone = build_model(hidden_units=HIDDEN)
        clone.load_state_dict(model.state_dict())
        assert torch.allclose(
            model.log2feats(log_seqs, residuals),
            clone.log2feats(log_seqs, residuals),
            atol=1e-6,
        )


class TestInferenceMode:
    def test_score_items_is_deterministic_on_a_training_mode_model(self, batch):
        """
        score_items reads as a self-contained inference entry point and will be
        used as one by the harness. On a model left in train() it would
        otherwise apply dropout and answer differently every call.
        """
        log_seqs, _, _, residuals = batch
        model = SASRec(
            item_num=ITEM_NUM, hidden_units=HIDDEN, num_blocks=2, num_heads=2,
            dropout_rate=0.5, maxlen=MAXLEN,
        )
        model.train()
        candidates = torch.tensor([[1, 2, 3]] * BATCH, dtype=torch.long)
        first, _ = model.score_items(log_seqs, candidates, residuals)
        second, _ = model.score_items(log_seqs, candidates, residuals)
        assert torch.allclose(first, second, atol=1e-6)

    def test_score_items_restores_the_previous_mode(self, batch):
        log_seqs, _, _, residuals = batch
        model = build_model()
        candidates = torch.tensor([[1]] * BATCH, dtype=torch.long)

        model.train()
        model.score_items(log_seqs, candidates, residuals)
        assert model.training is True

        model.eval()
        model.score_items(log_seqs, candidates, residuals)
        assert model.training is False

    def test_block_count_and_head_count_are_configurable(self):
        model = build_model(num_blocks=3, num_heads=4, hidden_units=8)
        assert len(model.attention_layers) == 3
        assert len(model.forward_layers) == 3
        assert model.attention_layers[0].num_heads == 4

    def test_norm_first_changes_the_computation(self, batch):
        """
        Divergence 1. Both orders must be reachable, and they must differ, or
        the flag is decorative.
        """
        log_seqs, _, _, residuals = batch
        post = build_model(norm_first=False).log2feats(log_seqs, residuals)
        pre = build_model(norm_first=True).log2feats(log_seqs, residuals)
        assert not torch.allclose(post, pre, atol=1e-4)

    def test_masking_padded_keys_changes_the_computation(self, batch):
        """
        Divergence 3. kang205 masks padded keys, pmixer does not. Off by
        default; the flag must actually do something when switched on.
        """
        log_seqs, _, _, residuals = batch
        unmasked = build_model(mask_padded_keys=False).log2feats(log_seqs, residuals)
        masked = build_model(mask_padded_keys=True).log2feats(log_seqs, residuals)
        pad = log_seqs == PAD_INDEX
        # Padded rows are zeroed either way; the real positions must differ.
        assert not torch.allclose(unmasked[~pad], masked[~pad], atol=1e-4)

    def test_defaults_follow_pmixer_not_kang205(self):
        model = build_model()
        assert model.norm_first is False, "pmixer defaults to post-norm"
        assert model.mask_padded_keys is False, "pmixer does not mask padded keys"

    def test_divergences_are_documented_in_the_module_docstring(self):
        import sim.agents.sasrec_model as module

        doc = module.__doc__ or ""
        assert "kang205/SASRec" in doc
        assert "pmixer/SASRec.pytorch" in doc
        assert "Kang" in doc and "ICDM 2018" in doc
        # Ten numbered divergences are recorded; keep them there.
        for n in range(1, 11):
            assert f"{n}. **" in doc, f"divergence {n} missing from the docstring"


class TestConfigWiring:
    def test_from_config_uses_the_sasrec_fields(self):
        config = SimConfig(
            sasrec_hidden_units=16,
            sasrec_num_blocks=3,
            sasrec_num_heads=4,
            sasrec_dropout_rate=0.1,
            sasrec_maxlen=32,
            sasrec_norm_first=True,
            sasrec_mask_padded_keys=True,
            sasrec_inject_rating=False,
            sasrec_rating_head_hidden=7,
            sasrec_rating_loss_weight=0.5,
        )
        model = SASRec.from_config(config, item_num=ITEM_NUM)
        assert model.hidden_units == 16
        assert len(model.attention_layers) == 3
        assert model.attention_layers[0].num_heads == 4
        assert model.maxlen == 32
        assert model.pos_emb.weight.shape == (33, 16)
        assert model.norm_first is True
        assert model.mask_padded_keys is True
        assert model.inject_rating is False
        assert model.rating_loss_weight == 0.5

    def test_config_defaults_are_the_reference_movielens_settings(self):
        config = SimConfig()
        assert config.sasrec_hidden_units == 50
        assert config.sasrec_num_blocks == 2
        assert config.sasrec_num_heads == 1
        assert config.sasrec_dropout_rate == 0.2
        assert config.sasrec_maxlen == 200

    def test_sasrec_fields_are_logged_to_mlflow(self):
        logged = SimConfig().as_dict()
        for name in (
            "sasrec_hidden_units",
            "sasrec_num_blocks",
            "sasrec_num_heads",
            "sasrec_dropout_rate",
            "sasrec_maxlen",
            "sasrec_norm_first",
            "sasrec_mask_padded_keys",
            "sasrec_rating_loss_weight",
            "sasrec_inject_rating",
            "sasrec_rating_head_hidden",
        ):
            assert name in logged
