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
        assert parsed.holdout_frac == defaults.holdout_frac
        assert parsed.min_ratings == defaults.min_ratings
        assert parsed.experiment_profile == defaults.experiment_profile
        assert parsed.num_rounds == defaults.num_rounds
        assert parsed.rec_list_size == defaults.rec_list_size
        assert parsed.accept_k == defaults.accept_k
        assert parsed.max_requests_per_round == defaults.max_requests_per_round
        assert parsed.mf_features == defaults.mf_features
        assert parsed.semantic_model == defaults.semantic_model
        assert parsed.agent_type == defaults.agent_type
        assert parsed.random_seed == defaults.random_seed
        assert parsed.force_rebuild_embeddings == defaults.force_rebuild_embeddings
