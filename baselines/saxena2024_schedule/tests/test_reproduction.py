import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model import InputError, Parameters, ScheduleModel, build_flight_set, fixture
from run import PROJECT_ROOT, run_once


class SaxenaPortTests(unittest.TestCase):
    def solve(self, name, *, formulation="paper", preprocess=True, **kwargs):
        fs = build_flight_set(fixture(name), params=Parameters(**kwargs), preprocess=preprocess)
        model = ScheduleModel(fs, formulation=formulation).build()
        result = model.solve(time_limit=30)
        return fs, model, result

    def test_arrival_has_source_minimum_times_and_residuals(self):
        fs, model, result = self.solve("arrival")
        self.assertEqual(result.status, "optimal")
        values = result.node_times["A1"]
        self.assertAlmostEqual(values["N"], 10.0, places=7)
        self.assertGreaterEqual(values["X"] - values["N"], 300.0 / 17.0 - 5.0 - 1e-7)
        self.assertGreaterEqual(values["c"] - values["R2"], 2.0 - 1e-7)
        self.assertLessEqual(max((row["violation"] for row in result.constraint_checks), default=0.0), 1e-6)
        self.assertEqual(result.summary["formulation"], "paper")
        expected = (
            7.0 * (300.0 / 17.0 - 5.0)
            + 10.0 * ((300.0 / 17.0 - 5.0) + (75.0 / 8.0 - 5.0) + 2.0 - (300.0 / 17.0 - 5.0))
            + 8.0 * (max(45.0 / 6.0 - 5.0, 1.0) + max(45.0 / 6.0 - 5.0, 1.0) + 1.0)
        ) / 10.0
        self.assertAlmostEqual(result.objective, expected, places=6)

    def test_departure_and_turnaround_constraints(self):
        _, _, departure = self.solve("departure")
        self.assertEqual(departure.status, "optimal")
        d = departure.node_times["D1"]
        self.assertGreaterEqual(d["G2"], 5.0 - 1e-7)
        self.assertGreaterEqual(d["R2"] - d["c"], 2.0 - 1e-7)
        expected_departure = (8.0 * (2.5 + 1.0) + 10.0 * (2.0 + (75.0 / 8.0 - 5.0)) + 7.0 * (300.0 / 17.0 - 5.0)) / 10.0
        self.assertAlmostEqual(departure.objective, expected_departure, places=6)

        fs, _, turnaround = self.solve("tat")
        self.assertEqual(turnaround.status, "optimal")
        t = turnaround.node_times["T1"]
        self.assertGreaterEqual(t["G1c1"] - t["G1c0"], 120.0 - 1e-7)
        self.assertAlmostEqual(fs.requests[0].dep_req_time, 10.0 + 12.647058823529413 + 4.375 + 2.0 + 120.0 + 6.0, places=5)

    def test_mixed_fixture_and_gate_capacity_rows(self):
        _, model, result = self.solve("mixed")
        self.assertEqual(result.status, "optimal")
        self.assertIn("GateQ", result.summary["constraint_rows_by_family"])
        self.assertIn("wake", result.summary["constraint_rows_by_family"])
        self.assertLessEqual(max((row["violation"] for row in result.constraint_checks), default=0.0), 1e-6)
        self.assertTrue(all(np.isfinite(value) for values in result.node_times.values() for value in values.values()))
        self.assertLessEqual(max((abs(value - round(value)) for value in result.integer_values.values()), default=0.0), 1e-6)
        self._check_route_headways(model, result)
        self._check_opposite_edge_intervals(model, result)
        _, cap_model, _ = self.solve("gate_capacity3")
        self.assertTrue(cap_model.y1_vars)
        self.assertGreater(cap_model._row_family_counts().get("GateCapacity2", 0), 0)
        _, _, cap_result = self.solve("gate_capacity3")
        self.assertEqual(cap_result.status, "optimal")
        intervals = [
            (values["G1c0"], values["G1c1"])
            for record_id, values in cap_result.node_times.items()
            if "G1c0" in values
        ]
        self.assertLessEqual(self._max_overlap(intervals), 3)

        two_tat = [
            {"id": "T1", "type": "tat", "class": "Small", "gate": "G1", "direction": "N", "arr_req_time": 0, "tat": 90},
            {"id": "T2", "type": "tat", "class": "Small", "gate": "G1", "direction": "E", "arr_req_time": 30, "tat": 90},
        ]
        fs_one = build_flight_set(two_tat, params=Parameters(gate_capacity=1))
        one_result = ScheduleModel(fs_one).solve(time_limit=30)
        self.assertEqual(one_result.status, "optimal")
        one_intervals = [(values["G1c0"], values["G1c1"]) for record_id, values in one_result.node_times.items() if "G1c0" in values]
        self.assertLessEqual(self._max_overlap(one_intervals), 1)

    def test_wake_formulation_exposes_ordered_pair_difference(self):
        requests = [
            {"id": "A1", "type": "arr", "class": "Small", "gate": "G1", "direction": "N", "arr_req_time": 10},
            {"id": "A2", "type": "arr", "class": "Ultra", "gate": "G3", "direction": "N", "arr_req_time": 40},
        ]
        fs = build_flight_set(requests, preprocess=False)
        legacy = ScheduleModel(fs, formulation="upstream-legacy").build()
        paper = ScheduleModel(fs, formulation="paper").build()
        self.assertEqual(sum(row.family == "wake" for row in legacy.builder.rows), 1)
        self.assertEqual(sum(row.family == "wake" for row in paper.builder.rows), 2)
        values = np.zeros(len(paper.builder.names), dtype=float)
        values[paper._t(fs.arrivals[0], "R2")] = 0.1
        values[paper._t(fs.arrivals[1], "R2")] = 0.0
        values[paper._y("R2", "A1", "A2")] = 0.0
        values[paper._y("R2", "A2", "A1")] = 1.0
        checks = paper.constraint_checks(values)
        reverse = [row for row in checks if row["family"] == "wake" and "A2,A1" in row["name"]]
        self.assertEqual(len(reverse), 1)
        self.assertGreater(reverse[0]["violation"], 0.0)

    def test_invalid_and_infeasible_inputs_are_reported(self):
        fs = build_flight_set(
            [{"id": "NEG", "type": "arr", "class": "Small", "gate": "G1", "direction": "N", "arr_req_time": -1}],
            preprocess=False,
        )
        result = ScheduleModel(fs).solve(time_limit=30)
        self.assertEqual(result.status, "infeasible")

    def test_source_ordered_arrival_preprocessing_and_propagation(self):
        requests = [
            {"id": "T0", "type": "tat", "class": "Small", "gate": "G1", "direction": "N", "arr_req_time": 0, "tat": 90},
            {"id": "A1", "type": "arr", "class": "Medium", "gate": "G3", "direction": "N", "arr_req_time": 0},
            {"id": "A2", "type": "arr", "class": "Jumbo", "gate": "G2", "direction": "E", "arr_req_time": 0},
        ]
        fs = build_flight_set(requests, preprocess=True)
        effective = {r.request_id: r.arr_req_time for r in fs.requests}
        original = {r.request_id: r.original_arr_req_time for r in fs.requests}
        self.assertEqual(original, {"T0": 0.0, "A1": 0.0, "A2": 0.0})
        self.assertGreater(effective["T0"], effective["A1"])
        self.assertGreaterEqual(abs(effective["T0"] - effective["A1"]), fs.params.d_sep_fix[0, 1] / (300.0 / ((300.0 / 17.0) - 5.0)) - 1e-8)
        self.assertGreaterEqual(abs(effective["T0"] - effective["A2"]), 6.5 - 1e-8)
        duration = fs.requests[0].dep_req_time - fs.requests[0].arr_req_time
        self.assertAlmostEqual(duration, 12.647058823529413 + 4.375 + 2.0 + 90.0 + 6.0, places=6)
        self.assertAlmostEqual(fs.requests[0].original_dep_req_time, 12.647058823529413 + 4.375 + 2.0 + 90.0 + 6.0, places=6)

    def test_input_is_not_mutated_and_outputs_are_serialized(self):
        requests = fixture("arrival")
        original = copy.deepcopy(requests)
        with tempfile.TemporaryDirectory() as directory:
            result = run_once(
                requests=requests,
                output=directory,
                seed=26,
                fixture_name="arrival",
                formulation="paper",
                time_limit=30,
                mip_rel_gap=None,
                preprocess=True,
            )
            self.assertEqual(requests, original)
            for filename in ("inputs.json", "result.json", "constraint_checks.json", "node_times.csv"):
                self.assertTrue((Path(directory) / filename).is_file())
            payload = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "optimal")
            self.assertEqual(payload["summary"]["evidence_type"], "newly_run")
        with self.assertRaises(InputError):
            run_once(
                requests=fixture("arrival"),
                output=PROJECT_ROOT / "outputs" / "rejected",
                seed=26,
                fixture_name="arrival",
                formulation="paper",
                time_limit=30,
                mip_rel_gap=None,
                preprocess=True,
            )

    @staticmethod
    def _max_overlap(intervals):
        events = []
        for start, end in intervals:
            events.append((round(float(start), 6), 1))
            events.append((round(float(end), 6), -1))
        active = maximum = 0
        for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
            active += delta
            maximum = max(maximum, active)
        return maximum

    def _check_route_headways(self, model, result):
        lengths = model.topology.edge_lengths(model.params)
        legs = [*model.flight_set.arrivals, *model.flight_set.departures]
        for left_index, left in enumerate(legs):
            for right in legs[left_index + 1 :]:
                for edge in left.edges:
                    if edge not in right.edges or edge not in model.topology.taxi_edges + model.topology.gate_edges + model.topology.direction_edges:
                        continue
                    u, v = edge.split("-", 1)
                    lt = result.node_times[left.record_id]
                    rt = result.node_times[right.record_id]
                    if lt[u] <= rt[u]:
                        table = model.params.d_sep_fix if edge in model.topology.direction_edges else model.params.d_sep_taxi
                        sep = table[left.class_id - 1, right.class_id - 1] / lengths[edge]
                        self.assertGreaterEqual(rt[u] + 1e-6, lt[u] + sep * (lt[v] - lt[u]))
                    else:
                        table = model.params.d_sep_fix if edge in model.topology.direction_edges else model.params.d_sep_taxi
                        sep = table[right.class_id - 1, left.class_id - 1] / lengths[edge]
                        self.assertGreaterEqual(lt[u] + 1e-6, rt[u] + sep * (rt[v] - rt[u]))

    def _check_opposite_edge_intervals(self, model, result):
        legs = [*model.flight_set.arrivals, *model.flight_set.departures]
        for left_index, left in enumerate(legs):
            for right in legs[left_index + 1 :]:
                for edge in left.edges:
                    u, v = edge.split("-", 1)
                    reverse = f"{v}-{u}"
                    if reverse not in right.edges:
                        continue
                    lt = result.node_times[left.record_id]
                    rt = result.node_times[right.record_id]
                    self.assertTrue(lt[v] <= rt[v] + 1e-6 or rt[u] <= lt[u] + 1e-6)


if __name__ == "__main__":
    unittest.main()
