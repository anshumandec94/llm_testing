from __future__ import annotations

import json
from dataclasses import replace

from sim.hpo import HPOConfig, run_hpo


class TestHPOConfig:
    def test_hpo_config_loads_from_json(self, tiny_config, tmp_path):
        base = replace(
            tiny_config,
            mlflow_tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test-hpo-load",
            force_rebuild_embeddings=False,
        )
        config_path = tmp_path / "hpo.json"
        config_path.write_text(
            json.dumps(
                {
                    "base_config": base.to_json_dict(),
                    "candidate_overrides": [
                        {"mf_features": 4, "mf_epochs": 2},
                        {"mf_features": 6, "mf_epochs": 3},
                    ],
                }
            )
        )

        loaded = HPOConfig.from_json_file(config_path)

        assert loaded.base_config.data_dir == base.data_dir
        assert loaded.base_config.embeddings_dir == base.embeddings_dir
        assert loaded.candidate_overrides[0]["mf_features"] == 4


class TestRunHPO:
    def test_run_hpo_uses_validation_and_writes_best_config(
        self, tiny_config, env, tmp_path, monkeypatch
    ):
        _ = env
        monkeypatch.chdir(tmp_path)
        base = replace(
            tiny_config,
            mlflow_tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test-hpo-run",
            force_rebuild_embeddings=False,
        )
        hpo_config = HPOConfig(
            base_config=base,
            candidate_overrides=[
                {
                    "mf_features": 4,
                    "mf_epochs": 2,
                    "mf_regularization": 0.05,
                    "mf_damping": 4.0,
                },
                {
                    "mf_features": 6,
                    "mf_epochs": 3,
                    "mf_regularization": 0.10,
                    "mf_damping": 5.0,
                },
            ],
        )

        result = run_hpo(hpo_config)

        assert len(result.candidate_results) == 2
        assert set(result.candidate_results["evaluation_split"]) == {"validation"}
        assert result.best_config.recommender_eval_split == "held_out"
        assert result.search_results_path.exists()
        assert result.best_config_path.exists()

        saved = json.loads(result.best_config_path.read_text())
        # Cache now stores only the tuned params (partial dict) for merge support.
        assert "mf_features" in saved
        assert saved["mf_features"] == result.best_config.mf_features
