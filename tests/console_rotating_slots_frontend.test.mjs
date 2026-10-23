import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

function consoleHarness(overrides = {}) {
  const elements = new Map();
  const element = () => ({ value: "", options: [], textContent: "", hidden: false, disabled: false });
  const context = vm.createContext({
    window: {}, console, Headers, URLSearchParams,
    document: { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); } },
    ...overrides,
  });
  const source = readFileSync(new URL("../terminal_marl/visualization/web/console.js", import.meta.url), "utf8")
    .replace(/^import Scene from .*;\r?\n/, "")
    .replace(/\binitialize\(\);\s*$/, "");
  const sceneSource = readFileSync(new URL("../terminal_marl/visualization/web/scene.js", import.meta.url), "utf8").replace("export default class Scene", "class Scene");
  vm.runInContext(`${sceneSource}\n${source}\nglobalThis.harness = {state, sameAircraft, discreteReplay, interpolateFrame, displayFrame, renderFlights, renderKpis, updateHistory, appendServerPayload, metricsForFrame, referenceAtTime, validateReplayPayload, formatSoc, requestJson};`, context);
  return context.harness;
}

function frame(time, extra = {}) {
  return { time, step_count: time, external_queue_size: 0, service_count: 0, completed_count: 0,
    rings: [], gates: [], landing_zones: [], uavs: [], recent_conflict_uids: [], terminal_events: [], telemetry: {}, ...extra };
}

test("paused flight rows remain connected until frame, selection or table controls change", () => {
  const elements = new Map();
  const node = (tagName = "DIV") => ({
    tagName, children: [], value: "", listeners: {}, attributes: {},
    classList: { values: new Set(), add(value) { this.values.add(value); }, contains(value) { return this.values.has(value); } },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; if (this.tagName === "SELECT") this.value = items[0]?.value ?? ""; },
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(name, handler) { this.listeners[name] = handler; },
  });
  const body = node("TBODY");
  elements.set("flightTable", { querySelector: () => body });
  elements.set("phaseFilter", node("SELECT"));
  const document = { createElement: (tag) => node(tag.toUpperCase()), getElementById(id) { if (!elements.has(id)) elements.set(id, node()); return elements.get(id); } };
  const h = consoleHarness({ document, Option: function(text, value) { this.textContent = text; this.value = value; } });
  const first = frame(0, { uavs: [
    { uid: 1, aircraft_id: "A-1", phase: "HOLD", soc: .8, speed: 10, position: [0, 0, 0] },
    { uid: 2, aircraft_id: "A-2", phase: "FINAL", soc: .3, speed: 12, position: [5, 0, 0] },
  ] });
  h.state.frames = [first];
  h.renderFlights(h.displayFrame());
  const initialRow = body.children[0];
  const phaseOption = elements.get("phaseFilter").children[1];
  h.renderFlights(h.displayFrame());
  assert.equal(body.children[0], initialRow);
  assert.equal(initialRow.tabIndex, 0);
  assert.equal(typeof initialRow.listeners.keydown, "function");

  h.state.selectedUid = 1; h.renderFlights(first);
  assert.notEqual(body.children[0], initialRow);
  assert.equal(body.children[0].attributes["aria-selected"], "true");
  assert.equal(elements.get("phaseFilter").children[1], phaseOption);
  elements.get("flightSearch").value = "A-2"; h.renderFlights(first);
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].children[0].textContent, "A-2");
  elements.get("flightSearch").value = "";
  elements.get("phaseFilter").value = "HOLD"; h.renderFlights(first);
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].children[0].textContent, "A-1");
  elements.get("phaseFilter").value = ""; elements.get("flightSort").value = "soc"; h.renderFlights(first);
  assert.equal(body.children[0].children[0].textContent, "A-2");

  const replacement = frame(0, { uavs: [{ ...first.uavs[0], speed: 22 }] });
  h.state.frames = [replacement]; // Different replay, same frame index and timestamp.
  h.renderFlights(h.displayFrame());
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].children[4].textContent, "22.0 m/s");
  const replacementRow = body.children[0];
  h.renderFlights(h.displayFrame()); assert.equal(body.children[0], replacementRow);

  h.state.frames = [first, frame(1, { uavs: first.uavs.map((uav) => ({ ...uav, position: [10, 0, 0] })) })];
  h.state.playing = true; h.state.playheadTime = .25;
  const interpolated = h.displayFrame();
  h.state.playheadTime = .75;
  assert.notEqual(h.displayFrame(), interpolated);
  assert.notEqual(h.displayFrame().uavs[0].position[0], interpolated.uavs[0].position[0]);
});

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

test("recorded energy and rolling flow values are selected-frame values, including zero and unknown", () => {
  const h = consoleHarness();
  const later = frame(480, { energy_diagnostics: { low_soc_threshold: .35, low_soc_count: 9, emergency_active_count: 6, emergency_landed_count: 0 },
    traffic_flow: { window_s: 60, nominal_rate_per_hour: 180, requested_per_hour: 180, admitted_per_hour: 120, landed_per_hour: 0 } });
  const earlier = frame(30, { energy_diagnostics: { low_soc_threshold: .35, low_soc_count: 1, emergency_active_count: 0, emergency_landed_count: null },
    traffic_flow: { window_s: 30, nominal_rate_per_hour: 180, requested_per_hour: 120, admitted_per_hour: 0, landed_per_hour: 0 } });
  h.state.frames = [earlier, later];
  h.state.currentIndex = 1;
  assert.equal(h.metricsForFrame(h.displayFrame()).emergencyActive, 6);
  assert.equal(h.metricsForFrame(later).emergencyLanded, 0);
  assert.equal(h.metricsForFrame(later).landedFlow, 0);
  h.state.currentIndex = 0;
  const values = h.metricsForFrame(h.displayFrame());
  assert.equal(values.lowSoc, 1); assert.equal(values.emergencyActive, 0); assert.equal(values.emergencyLanded, null);
  assert.equal(values.flowWindow, 30); assert.equal(values.nominalFlow, 180); assert.equal(values.requestedFlow, 120);
  assert.equal(values.admittedFlow, 0);
  assert.equal(h.metricsForFrame(frame(0, { traffic_flow: { window_s: 0, landed_per_hour: 0 } })).landedFlow, null);
  const unknown = h.metricsForFrame({ time: 0 });
  for (const key of ["lowSoc", "emergencyActive", "emergencyLanded", "nominalFlow", "requestedFlow", "admittedFlow", "landedFlow"]) assert.equal(unknown[key], null, key);
});

test("legacy current counts use flags and exclude ground/frozen records; cumulative counts require touchdown", () => {
  const h = consoleHarness();
  const f = frame(15, { uavs: [
    { phase: "HOLD", soc: .3, low_soc: true, emergency_landing: true },
    { phase: "HOLD", energy_alert: "low_soc", emergency_landing: false },
    { phase: "HOLD", energy_alert: "normal", emergency_landing: false },
    ...["LANDED_SERVICE", "CLEARED", "ENERGY_MISSION_FAILED", "CENSORED_DUE_TO_FAILURE"].map((status) => ({ phase: "HOLD", status, low_soc: true, emergency_landing: true })),
  ], terminal_events: [{ type: "emergency_declared", time: 5, emergency: true },
    { outcome: "touchdown", time: 10, emergency: true }, { outcome: "touchdown", time: 12, emergency: false },
    { outcome: "touchdown", time: 20, emergency: true }] });
  const values = h.metricsForFrame(f);
  assert.equal(values.lowSoc, 2); assert.equal(values.emergencyActive, 1); assert.equal(values.emergencyLanded, 1);
  assert.equal(values.threshold, null);
  assert.equal(h.metricsForFrame(frame(0, { uavs: [{ phase: "HOLD", soc: .1 }] })).lowSoc, null);
  assert.equal(h.metricsForFrame(frame(15, { terminal_events: [{ outcome: "touchdown", time: 10 }] })).emergencyLanded, null);
  assert.equal(h.metricsForFrame(frame(15, { telemetry: { L_emergency: 4 } })).emergencyLanded, 4);
  assert.equal(h.metricsForFrame(frame(15, { telemetry: { emergency_landed: 0, L_emergency: 4 } })).emergencyLanded, 0);
  assert.equal(h.metricsForFrame(frame(0)).lowSoc, 0);
  assert.equal(h.metricsForFrame(frame(0)).emergencyLanded, null);
  assert.equal(h.metricsForFrame(frame(0, { console_replay_kind: "rotating_slots" })).emergencyLanded, 0);
  assert.equal(h.metricsForFrame(frame(0, { console_replay_kind: "rotating_slots", energy_diagnostics: { emergency_landed_count: null } })).emergencyLanded, null);
  h.state.metadata = { replay_kind: "rotating_slots" };
  assert.equal(h.metricsForFrame(frame(0)).emergencyLanded, null);
});

test("KPI labels retain units, partial window and unknown values after seeking", () => {
  const elements = new Map();
  const h = consoleHarness({ document: { getElementById(id) { if (!elements.has(id)) elements.set(id, {}); return elements.get(id); } } });
  h.renderKpis(frame(480, { energy_diagnostics: { low_soc_threshold: .35, low_soc_count: 8, emergency_active_count: 6, emergency_landed_count: 0 }, traffic_flow: { window_s: 60, nominal_rate_per_hour: 180, admitted_per_hour: 120, landed_per_hour: 0 } }));
  assert.equal(elements.get("kpiLowSocLabel").textContent, "低电量 ≤35%");
  assert.equal(elements.get("kpiEmergencyActive").textContent, "当前应急 6");
  assert.equal(elements.get("kpiEmergencyLanded").textContent, "0");
  assert.equal(elements.get("kpiDemand").textContent, "180");
  assert.equal(elements.get("kpiLandedFlow").textContent, "0.0");
  h.renderKpis(frame(10, { traffic_flow: { window_s: 10, landed_per_hour: 0 } }));
  assert.equal(elements.get("kpiFlowWindow").textContent, "近10.0 s / 起始窗口");
  assert.equal(elements.get("kpiDemand").textContent, "—");
  h.renderKpis(null);
  assert.equal(elements.get("kpiEmergencyActive").textContent, "当前应急 —");
  assert.equal(elements.get("kpiEmergencyLanded").textContent, "—");
});

test("emergency history crosses approach/descent phases, splits gaps and rewinds without future points", () => {
  const h = consoleHarness();
  const uav = (phase, x, extra = {}) => ({ uid: 7, phase, position: [x, 0, 20 - x], committed: true, trajectory_kind: "FINAL_EMERGENCY", ...extra });
  h.state.frames = [frame(0, { uavs: [uav("FINAL_EMERGENCY", 0)] }),
    frame(1, { uavs: [uav("VERTICAL_DESCENT", 1)] }),
    frame(2, { uavs: [uav("LANDED_SERVICE", 2)] }), frame(3),
    frame(4, { uavs: [uav("FINAL_EMERGENCY", 4)] }), frame(5)];
  let points = h.updateHistory(5).get("7:emergency");
  assert.deepEqual(JSON.parse(JSON.stringify(points)), [[0, 0, 20], [1, 0, 19], null, [4, 0, 16], null]);
  points = h.updateHistory(1).get("7:emergency");
  assert.deepEqual(JSON.parse(JSON.stringify(points)), [[0, 0, 20], [1, 0, 19]]);
  h.state.frames = Array.from({ length: 905 }, (_, index) => frame(index, { uavs: [uav("FINAL_EMERGENCY", index)] }));
  h.state.historyCache = { index: -1, map: new Map() };
  assert.equal(h.updateHistory(904).get("7:emergency").length, 900);
});
