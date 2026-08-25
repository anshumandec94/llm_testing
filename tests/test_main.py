from __future__ import annotations

from sim.config import SimConfig
from main import parse_args


class TestParseArgs:
    def test_cli_defaults_match_simconfig_defaults(self):
        parsed = parse_args([])
        defaults = SimConfig()

        assert parsed.data_dir == defaults.data_dir
        assert parsed.embeddings_dir == defaults.embeddings_dir
        assert parsed.mlflow_tracking_uri == defaults.mlflow_tracking_uri
        assert parsed.experiment_name == defaults.experiment_name
        assert parsed.eval_user_frac == defaults.eval_user_frac
        assert parsed.validation_frac == defaults.validation_frac
        assert parsed.holdout_frac == defaults.holdout_frac
        assert parsed.min_ratings == defaults.min_ratings
        assert parsed.experiment_profile == defaults.experiment_profile
        assert parsed.recommender_eval_split == defaults.recommender_eval_split
        assert parsed.num_rounds == defaults.num_rounds
        assert parsed.rec_list_size == defaults.rec_list_size
        assert parsed.accept_k == defaults.accept_k
        assert parsed.max_requests_per_round == defaults.max_requests_per_round
        assert parsed.mf_features == defaults.mf_features
        assert parsed.mf_epochs == defaults.mf_epochs
        assert parsed.mf_regularization == defaults.mf_regularization
        assert parsed.mf_damping == defaults.mf_damping
        assert parsed.semantic_model == defaults.semantic_model
        assert parsed.agent_type == defaults.agent_type
        assert parsed.agent_types == defaults.agent_types
        assert parsed.agent_type_proportions == defaults.agent_type_proportions
        assert parsed.agent_assignment_mode == defaults.agent_assignment_mode
        assert parsed.random_seed == defaults.random_seed
        assert parsed.force_rebuild_embeddings == defaults.force_rebuild_embeddings

    def test_cache_keys_change_when_split_or_mf_params_change(self):
        base = SimConfig()
        changed_split = SimConfig(validation_frac=base.validation_frac + 0.05)
        changed_mf = SimConfig(mf_epochs=base.mf_epochs + 1)

        assert base.split_cache_key() != changed_split.split_cache_key()
        assert (
            base.platform_factor_cache_key()
            != changed_mf.platform_factor_cache_key()
        )

    def test_agent_type_accepts_associative_baseline_alias(self):
        parsed = parse_args(["--agent_type", "associative_baseline"])
        assert parsed.agent_type == "associative_baseline"

    def test_agent_type_accepts_traditional_agent_variants(self):
        assert parse_args(["--agent_type", "residual_profile"]).agent_type == "residual_profile"
        assert parse_args(["--agent_type", "item_item"]).agent_type == "item_item"

    def test_assignment_args_accept_mixed_population_configuration(self):
        parsed = parse_args(
            [
                "--agent_types",
                "associative,item_item",
                "--agent_type_proportions",
                "0.25,0.75",
                "--agent_assignment_mode",
                "one_per_agent_type",
            ]
        )
        assert parsed.agent_types == ["associative", "item_item"]
        assert parsed.agent_type_proportions == [0.25, 0.75]
        assert parsed.agent_assignment_mode == "one_per_agent_type"
