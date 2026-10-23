import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

function consoleHarness(overrides = {}) {
  const elements = new Map();
  const element = () => ({ value: "", options: [], textContent: "", hidden: false, disabled: false });
  const context = vm.createContext({
    window: {}, console, Headers, URLSearchParams, ...overrides,
    document: { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); } },
  });
  const source = readFileSync(new URL("../terminal_marl/visualization/web/console.js", import.meta.url), "utf8")
    .replace(/^import Scene from .*;\r?\n/, "")
    .replace(/\binitialize\(\);\s*$/, "");
  vm.runInContext(`${source}\nglobalThis.harness = {state, sameAircraft, discreteReplay, interpolateFrame, appendServerPayload, metricsForFrame, referenceAtTime, validateReplayPayload, formatSoc, requestJson};`, context);
  return context.harness;
}

function frame(time, extra = {}) {
  return { time, step_count: time, external_queue_size: 0, service_count: 0, completed_count: 0,
    rings: [], gates: [], landing_zones: [], uavs: [], recent_conflict_uids: [], terminal_events: [], telemetry: {}, ...extra };
}

test("cross-run identity uses original aircraft strings before compatibility IDs", () => {
  const h = consoleHarness();
  assert.equal(h.sameAircraft({ aircraft_id: "A-17", arrival_id: 1 }, { aircraft_id: "A-17", arrival_id: 99 }), true);
  assert.equal(h.sameAircraft({ aircraft_id: "A-17", arrival_id: 1 }, { aircraft_id: "B-17", arrival_id: 1 }), false);
  assert.equal(h.sameAircraft({ aircraft_id: "0017" }, { aircraft_id: "17" }), false);
  assert.equal(h.sameAircraft({ aircraft_id: "A-17", arrival_id: 17 }, { arrival_id: 17 }), false);
  assert.equal(h.sameAircraft({ arrival_id: 17 }, { arrival_id: "17" }), true);
});

test("rotating replay keeps original poses; legacy replay still interpolates", () => {
  const h = consoleHarness();
  const first = frame(0, { uavs: [{ uid: 1, phase: "HOLD", position: [0, 0, 0] }] });
  const second = frame(1, { uavs: [{ uid: 1, phase: "HOLD", position: [10, 10, 0] }] });
  assert.equal(h.interpolateFrame(first, second, .5).uavs[0].position[0], 5);
  first.console_replay_kind = "rotating_slots";
  assert.equal(h.interpolateFrame(first, second, .5), first);
  delete first.console_replay_kind;
  h.state.metadata = { run_metadata: { replay_kind: "rotating_slots", interpolation: "none" } };
  assert.equal(h.interpolateFrame(first, second, .5), first);
});

test("server replay generation and additional pages remain at first frame paused", () => {
  const h = consoleHarness();
  h.state.generation = 1; h.state.frames = [frame(90)]; h.state.currentIndex = 0;
  h.state.playing = true; h.state.viewMode = "live"; h.state.liveFollow = true;
  h.appendServerPayload({ generation: 2, status: "replay", frames: [frame(0), frame(1)], metadata: { replay_kind: "rotating_slots" }, next_cursor: 2 });
  assert.equal(h.state.currentIndex, 0); assert.equal(h.state.playheadTime, 0);
  assert.equal(h.state.playing, false); assert.equal(h.state.liveFollow, false);
  h.appendServerPayload({ generation: 2, status: "replay", frames: [frame(2)], next_cursor: 3 });
  assert.equal(h.state.currentIndex, 0); assert.equal(h.state.frames.length, 3);
  h.appendServerPayload({ generation: 3, status: "running", metadata: {}, frames: [frame(7), frame(8)] });
  h.state.viewMode = "live"; h.state.liveFollow = true;
  h.appendServerPayload({ generation: 3, status: "running", frames: [frame(9)] });
  assert.equal(h.state.currentIndex, 2);
});

test("recorded KPI and reference time alignment preserve their semantics", () => {
  const h = consoleHarness();
  const f = frame(240, { active_count: 2, service_count: 1, uavs: [{ soc: .5 }, { soc: null }, {}], rotating_slots: { upstream_waiting_count: 7 } });
  assert.equal(h.metricsForFrame(f).active, 2); assert.equal(h.metricsForFrame(f).queue, 7);
  assert.equal(h.metricsForFrame(f).minSoc, .5); assert.equal(h.formatSoc(null), "—");
  const ref = frame(239.9999999999999);
  h.state.reference = { frames: [ref] };
  assert.equal(h.referenceAtTime(240), ref); assert.equal(h.referenceAtTime(241), null);
});

test("unknown terminal SOC is allowed only for explicit unavailable rotating records", () => {
  const h = consoleHarness();
  const f = frame(0, { console_replay_kind: "rotating_slots", terminal_events: [{ time: 0, soc: null, soc_source: "unavailable" }] });
  assert.equal(h.validateReplayPayload({ frames: [f] }).frames.length, 1);
  delete f.console_replay_kind;
  assert.throws(() => h.validateReplayPayload({ frames: [f] }), /soc/);
});

test("API errors expose the descriptive server message", async () => {
  const message = "replay cannot be read: D:/missing/replay.json";
  const h = consoleHarness({ fetch: async () => ({ ok: false, status: 400, text: async () => JSON.stringify({ error: "invalid_request", message }) }) });
  await assert.rejects(h.requestJson("/replay", { method: "POST", body: "{}" }), (error) => error.message === message && error.status === 400);
});
