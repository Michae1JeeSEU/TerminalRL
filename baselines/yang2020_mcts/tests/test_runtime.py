import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from baselines.yang2020_mcts import runtime


class RuntimeContractTests(unittest.TestCase):
    def test_source_snapshot_and_runtime_guard(self):
        self.assertEqual(len(runtime.SOURCE_SHA), 40)
        self.assertFalse((runtime.UPSTREAM / ".git").exists())
        for module_name in ("nodes_multi", "nodesHexSecGatePlus"):
            text = (runtime.MCTS_DIR / f"{module_name}.py").read_text(encoding="utf-8")
            self.assertEqual(text.count("if self.state.init_action == 'random':"), 1)
            patched, count = runtime._patched_node_source(runtime.MCTS_DIR / f"{module_name}.py")
            self.assertEqual(count, 1)
            self.assertIn(
                "if isinstance(self.state.init_action, str) and self.state.init_action == 'random':",
                patched,
            )
            self.assertNotIn("isinstance(self.state.init_action, str)", text)

    def test_source_case_contract(self):
        self.assertEqual(runtime.CASES["crossing"].start_counter, 1)
        self.assertEqual(runtime.CASES["hex"].start_counter, 0)
        self.assertEqual(runtime.CASES["hex_two_stage"].start_counter, 0)
        self.assertEqual(runtime.CASES["crossing"].threshold_multiplier, 3)
        self.assertEqual(runtime.CASES["hex"].threshold_multiplier, 2)
        self.assertEqual(runtime.CASES["hex_two_stage"].threshold_multiplier, 5)
        self.assertEqual(runtime.PLAN_PERIOD, 5)

    def test_output_must_be_external(self):
        with self.assertRaises(ValueError):
            runtime._assert_external_output(runtime.WORKSPACE_ROOT / "outputs")

    def test_crossing_stop_clock_uses_incremented_source_counter(self):
        spec = runtime.CASES["crossing"]
        self.assertEqual(100 + spec.start_counter, 101)
        self.assertGreater(101, spec.natural_after_empty_steps)

    def test_sector_cases_use_source_cadence_and_thresholds(self):
        self.assertEqual(runtime.CASES["hex"].start_counter, 0)
        self.assertEqual(runtime.CASES["hex_two_stage"].start_counter, 0)
        self.assertEqual(runtime.CASES["hex"].threshold_multiplier, 2)
        self.assertEqual(runtime.CASES["hex_two_stage"].threshold_multiplier, 5)
        self.assertTrue(runtime.CASES["hex_two_stage"].two_stage)

    def test_crossing_finite_artifacts_and_five_step_planning(self):
        # One rollout/depth/frame keeps this contract test bounded while still
        # executing the source MCTS decision path.
        with tempfile.TemporaryDirectory() as tmp:
            result = runtime.run_case(
                runtime.RunOptions(
                    case="crossing",
                    seed=17,
                    max_steps=6,
                    output=Path(tmp),
                    full_simulations=1,
                    lite_simulations=1,
                    search_depth=1,
                    lite_search_depth=1,
                    simulate_frame=1,
                )
            )
            self.assertEqual(result.steps, 6)
            self.assertTrue(result.truncated)
            self.assertGreater(result.mcts_decisions, 0)
            metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["plan_period"], 5)
            self.assertEqual(metadata["original_config"]["full_simulations"], 100)
            self.assertEqual(metadata["overrides"]["full_simulations"], 1)
            self.assertEqual(metadata["compatibility_transform"]["upstream/MCTS/nodes_multi.py"], 1)
            with (Path(tmp) / "actions.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual({int(row["step"]) for row in rows}, {5})
            self.assertEqual(len(rows), result.mcts_decisions)

    def test_crossing_decision_matches_corrected_source_loop(self):
        # Build an identical state and compare the wrapper's planning helper to
        # the original entry-point loop with only the required string guard.
        spec = runtime.CASES["crossing"]
        modules = runtime._load_upstream(spec)
        config = modules["Config"]
        config.simulate_frame = 1
        np.random.seed(23)
        random.seed(23)
        env = modules["Env"](23, debug=False)
        observation = env.reset()
        info = None
        for _ in range(5):
            observation, _reward, _done, info = env.step({})

        options = runtime.RunOptions(
            case="crossing",
            seed=23,
            max_steps=6,
            output=Path(tempfile.gettempdir()),
            full_simulations=1,
            lite_simulations=1,
            search_depth=1,
            lite_search_depth=1,
        )

        def reference_plan():
            ob, ids = observation
            actions = np.ones(ob.shape[0], dtype=np.int32)
            by_id = {}
            for index in range(ob.shape[0]):
                state = modules["MultiAircraftState"](
                    state=ob, index=index, init_action=actions
                )
                root = modules["MultiAircraftNode"](state=state)
                mcts = modules["MCTS"](root)
                full = info[index] < 3 * config.minimum_separation
                if full:
                    node = mcts.best_action(1, 1)
                else:
                    node = mcts.best_action(1, 1)
                actions[index] = node.state.prev_action[index]
                by_id[ids[index]] = node.state.prev_action[index]
            return by_id

        np.random.seed(91)
        expected = reference_plan()
        np.random.seed(91)
        trace = []
        actual = runtime._plan_crossing(
            modules, env, observation, info, 5, options, trace, config
        )
        self.assertEqual({key: int(value) for key, value in expected.items()},
                         {key: int(value) for key, value in actual.items()})
        self.assertEqual({row["step"] for row in trace}, {5})
        self.assertEqual(len(trace), len(expected))

    def _sector_state_after_five_steps(self, case):
        spec = runtime.CASES[case]
        modules = runtime._load_upstream(spec)
        modules["Config"].simulate_frame = 1
        np.random.seed(31)
        random.seed(31)
        env = modules["Env"](31, debug=False)
        observation = env.reset()
        info = None
        planning_step = None
        for step in range(1, 81):
            observation, _reward, _done, info = env.step({}, False)
            if step % runtime.PLAN_PERIOD == 0:
                if any(len(value[1]) for value in observation.values()):
                    planning_step = step
                    break
        self.assertIsNotNone(planning_step)
        return modules, env, observation, info, planning_step

    def test_hex_decision_matches_source_sector_loop(self):
        modules, env, observation, info, planning_step = self._sector_state_after_five_steps("hex")
        config = modules["Config"]
        options = runtime.RunOptions(
            case="hex", seed=31, output=Path(tempfile.gettempdir()),
            full_simulations=1, lite_simulations=1, search_depth=1, lite_search_depth=1,
        )

        def reference_plan():
            import numpy as np
            by_id = {}
            for sector_id in range(7):
                ob, ids, goals = observation[sector_id]
                actions = np.ones(ob.shape[0], dtype=np.int32)
                for index in range(len(ids)):
                    state = modules["MultiAircraftState"](
                        state=ob, index=index, init_action=actions,
                        sector_id=sector_id, goal_exit_id=goals[index]
                    )
                    root = modules["MultiAircraftNode"](state=state)
                    mcts = modules["MCTS"](root)
                    full = info[ids[index]] < 2 * config.minimum_separation
                    node = mcts.best_action(1, 1)
                    actions[index] = node.state.prev_action[index]
                    by_id[ids[index]] = node.state.prev_action[index]
                    self.assertIn(full, (True, False))
            return by_id

        np.random.seed(97)
        expected = reference_plan()
        np.random.seed(97)
        trace = []
        actual = runtime._plan_hex(modules, env, observation, info, planning_step, options, trace, config)
        self.assertEqual({k: int(v) for k, v in expected.items()},
                         {k: int(v) for k, v in actual.items()})
        self.assertEqual({row["step"] for row in trace}, {planning_step})

    def test_hex_two_stage_decision_matches_source_loop(self):
        modules, env, observation, info, planning_step = self._sector_state_after_five_steps("hex_two_stage")
        config = modules["Config"]
        options = runtime.RunOptions(
            case="hex_two_stage", seed=31, output=Path(tempfile.gettempdir()),
            full_simulations=1, lite_simulations=1, search_depth=1, lite_search_depth=1,
        )

        def reference_plan():
            import numpy as np
            by_id = {}
            for sector_id in range(7):
                (ob_high_in, id_high, goals_high, ob_high_out,
                 ob_in, ids, goals, ob_out) = observation[sector_id]
                ob_high = np.concatenate([ob_high_in, ob_high_out])
                actions_high = np.ones(ob_high.shape[0], dtype=np.int32)
                for index in range(len(id_high)):
                    state = modules["MultiAircraftState"](
                        state=ob_high, index=index, init_action=actions_high,
                        sector_id=sector_id, goal_exit_id=goals_high[index]
                    )
                    root = modules["MultiAircraftNode"](state=state)
                    node = modules["MCTS"](root).best_action(1, 1)
                    actions_high[index] = node.state.prev_action[index]
                    by_id[id_high[index]] = node.state.prev_action[index]
                ob = np.concatenate([ob_in, ob_high_in, ob_high_out, ob_out])
                actions = np.ones(ob.shape[0], dtype=np.int32)
                actions[len(ids):len(ids) + actions_high.shape[0]] = actions_high
                for index in range(len(ids)):
                    state = modules["MultiAircraftState"](
                        state=ob, index=index, init_action=actions,
                        sector_id=sector_id, goal_exit_id=goals[index]
                    )
                    root = modules["MultiAircraftNode"](state=state)
                    node = modules["MCTS"](root).best_action(1, 1)
                    actions[index] = node.state.prev_action[index]
                    by_id[ids[index]] = node.state.prev_action[index]
            return by_id

        np.random.seed(97)
        expected = reference_plan()
        np.random.seed(97)
        trace = []
        actual = runtime._plan_two_stage(
            modules, env, observation, info, planning_step, options, trace, config
        )
        self.assertEqual({k: int(v) for k, v in expected.items()},
                         {k: int(v) for k, v in actual.items()})
        self.assertEqual({row["step"] for row in trace}, {planning_step})


if __name__ == "__main__":
    unittest.main()
