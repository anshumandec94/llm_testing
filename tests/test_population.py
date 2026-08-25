from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim.persona import build_population_for_assignments
from sim.population import build_user_assignments


class TestUserAssignments:
    def test_one_to_one_respects_requested_agent_type_mix(self, tiny_config, env):
        cfg = replace(
            tiny_config,
            agent_types=["associative", "item_item"],
            agent_type_proportions=[0.0, 1.0],
            agent_assignment_mode="one_to_one",
        )
        rng = np.random.default_rng(cfg.random_seed)

        assignments = build_user_assignments(cfg, env, rng)

        assert len(assignments) == len(env.eval_users)
        assert {assignment.sim_user_id for assignment in assignments} == {
            int(uid) for uid in env.eval_users
        }
        assert {assignment.base_user_id for assignment in assignments} == {
            int(uid) for uid in env.eval_users
        }
        assert {assignment.agent_type for assignment in assignments} == {"item_item"}

    def test_one_per_agent_type_replicates_each_base_user(self, tiny_config, env):
        cfg = replace(
            tiny_config,
            agent_types=["associative", "item_item"],
            agent_assignment_mode="one_per_agent_type",
        )
        rng = np.random.default_rng(cfg.random_seed)

        assignments = build_user_assignments(cfg, env, rng)

        assert len(assignments) == len(env.eval_users) * 2
        for uid in env.eval_users:
            base_assignments = [
                assignment
                for assignment in assignments
                if assignment.base_user_id == int(uid)
            ]
            assert len(base_assignments) == 2
            assert {assignment.agent_type for assignment in base_assignments} == {
                "associative",
                "item_item",
            }
            assert len({assignment.sim_user_id for assignment in base_assignments}) == 2

    def test_assignment_aware_population_clones_base_persona_state(self, tiny_config, env):
        cfg = replace(
            tiny_config,
            agent_types=["associative", "item_item"],
            agent_assignment_mode="one_per_agent_type",
        )
        rng = np.random.default_rng(cfg.random_seed)
        assignments = build_user_assignments(cfg, env, rng)

        population = build_population_for_assignments(cfg, env, assignments, rng)

        base_user_id = int(env.eval_users[0])
        replicas = [
            population[assignment.sim_user_id]
            for assignment in assignments
            if assignment.base_user_id == base_user_id
        ]
        assert len(replicas) == 2
        assert replicas[0] is not replicas[1]
        assert replicas[0].user_id != replicas[1].user_id
        assert replicas[0].archetype == replicas[1].archetype
        assert np.allclose(replicas[0].pref_vector, replicas[1].pref_vector)
