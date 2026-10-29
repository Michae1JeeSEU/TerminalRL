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
  vm.runInContext(`${sceneSource}\n${source}\nglobalThis.harness = {state, sameAircraft, discreteReplay, interpolateFrame, displayFrame, renderFlights, renderEvents, renderDetails, renderResources, renderKpis, updateHistory, appendServerPayload, metricsForFrame, referenceAtTime, validateReplayPayload, formatSoc, requestJson, parseLocatorUid, locateUid, wireUidLocator, useDetailsOnlyUi: () => { renderUI = () => renderDetails(displayFrame()); }};`, context);
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
  elements.get("flightSearch").value = "UID 2"; h.renderFlights(first);
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].children[0].textContent, "UID 2");
  elements.get("flightSearch").value = "";
  elements.get("phaseFilter").value = "HOLD"; h.renderFlights(first);
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].children[0].textContent, "UID 1");
  elements.get("phaseFilter").value = ""; elements.get("flightSort").value = "soc"; h.renderFlights(first);
  assert.equal(body.children[0].children[0].textContent, "UID 2");

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

function uidUiHarness() {
  const elements = new Map();
  const ctx = new Proxy({}, { get: () => () => {} });
  const node = () => ({
    children: [], value: "", textContent: "", listeners: {}, attributes: {},
    classList: { values: new Set(), add(value) { this.values.add(value); }, toggle(value, on) { on ? this.values.add(value) : this.values.delete(value); } },
    append(...items) { this.children.push(...items); }, replaceChildren(...items) { this.children = items; },
    setAttribute(name, value) { this.attributes[name] = value; }, addEventListener(name, handler) { this.listeners[name] = handler; }, getBoundingClientRect: () => ({ width: 200, height: 60 }), getContext: () => ctx,
    querySelector() { if (!this.body) this.body = node(); return this.body; },
  });
  const document = { createElement: node, querySelector: (selector) => document.getElementById(selector), getElementById(id) { if (!elements.has(id)) elements.set(id, node()); return elements.get(id); } };
  const h = consoleHarness({ document, Option: function(text, value) { this.textContent = text; this.value = value; } });
  return { h, get: document.getElementById };
}

test("locator accepts complete nonnegative safe UIDs and rejects partial or unsafe input", () => {
  const h = consoleHarness();
  for (const [input, expected] of [["0", 0], [" 17 ", 17], [" uId 17 ", 17], ["UID 0", 0], ["9007199254740991", Number.MAX_SAFE_INTEGER]]) assert.equal(h.parseLocatorUid(input), expected);
  for (const input of ["", "  ", "UID", "UID17", "17abc", "-1", "+1", "1.0", "1e2", "Infinity", "NaN", "9007199254740992", "UID -1"]) assert.equal(h.parseLocatorUid(input), null, input);
});

test("locator selects exact UID zero, opens details and renders the resized frame before focus without changing playback", () => {
  for (const playing of [false, true]) {
    const { h, get } = uidUiHarness(); h.useDetailsOnlyUi();
    const current = frame(12, { console_replay_kind: "rotating_slots", uavs: [
      { uid: null, position: [1, 2, 3] }, { uid: 10, position: [10, 10, 10] }, { uid: 0, position: [20, 30, 40], low_soc: true },
    ] });
    h.state.frames = [frame(0), current, frame(20)]; h.state.currentIndex = 1;
    h.state.playheadTime = 12.5; h.state.playbackSpeed = 4; h.state.playing = playing;
    h.state.selectedUid = 10; h.state.selectedEvent = { uid: 10 }; h.state.followUid = 10;
    get("detailsRail").hidden = true;
    const calls = [];
    h.state.scene = {
      setFrame(f, options) {
        assert.equal(get("detailsRail").hidden, false);
        assert.equal(get(".workspace").classList.values.has("details-open"), true);
        assert.equal(h.state.followUid, null); assert.equal(options.selectedUid, 0);
        calls.push(["render", f]);
      },
      focusOn(uid, f) { assert.equal(calls.at(-1)[0], "render"); calls.push(["focus", uid, f]); return true; },
    };
    assert.equal(h.locateUid("UID 0"), true);
    assert.equal(calls.length, 2); assert.equal(calls[0][1], current);
    assert.equal(calls[1][1], 0); assert.equal(calls[1][2], current);
    assert.equal(h.state.selectedUid, 0); assert.equal(h.state.selectedEvent, null); assert.equal(h.state.followUid, null);
    assert.equal(get("selectedTitle").textContent, "UID 0"); assert.equal(get("uidLocateStatus").textContent, "已定位 UID 0");
    assert.equal(h.state.currentIndex, 1); assert.equal(h.state.playheadTime, 12.5);
    assert.equal(h.state.playing, playing); assert.equal(h.state.playbackSpeed, 4);
  }
});

test("invalid, absent and unpositioned locator results leave selection, camera and playback untouched", () => {
  const { h, get } = uidUiHarness();
  const selectedEvent = { uid: 3 };
  Object.assign(h.state, { selectedUid: 3, selectedEvent, followUid: 3, currentIndex: 1, playheadTime: 12, playing: true, playbackSpeed: 2 });
  get("detailsRail").hidden = true;
  const calls = [];
  h.state.scene = { setFrame: () => calls.push("render"), focusOn: () => calls.push("focus") };
  const unchanged = () => {
    assert.equal(h.state.selectedUid, 3); assert.equal(h.state.selectedEvent, selectedEvent); assert.equal(h.state.followUid, 3);
    assert.equal(h.state.currentIndex, 1); assert.equal(h.state.playheadTime, 12); assert.equal(h.state.playing, true); assert.equal(h.state.playbackSpeed, 2);
    assert.equal(get("detailsRail").hidden, true); assert.deepEqual(calls, []);
  };
  for (const input of ["", "17junk", "-17", "1.7", "9007199254740992"]) { assert.equal(h.locateUid(input), false); assert.match(get("uidLocateStatus").textContent, /非负整数 UID/); unchanged(); }
  assert.equal(h.locateUid("17"), false); assert.match(get("uidLocateStatus").textContent, /没有场景帧/); unchanged();
  h.state.frames = [frame(0, { uavs: [{ uid: 17, position: [1, 2, 3] }] }), frame(12, { uavs: [{ uid: 170, position: [1, 2, 3] }] })];
  h.state.reference = { frames: [frame(12, { uavs: [{ uid: 17, position: [1, 2, 3] }] })] };
  assert.equal(h.locateUid("17"), false); assert.match(get("uidLocateStatus").textContent, /当前帧未找到 UID 17/); unchanged();
  for (const position of [undefined, [1, 2], [1, null, 3], [1, NaN, 3], [1, Infinity, 3]]) {
    h.state.frames[1] = frame(12, { uavs: [{ uid: 17, position }] });
    assert.equal(h.locateUid("17"), false); assert.match(get("uidLocateStatus").textContent, /没有有效位置/); unchanged();
  }
});

test("locator form binds native submit and keeps input keyboard events away from playback shortcuts", () => {
  const { h, get } = uidUiHarness();
  h.wireUidLocator(); get("uidLocateInput").value = "UID 17";
  let prevented = false, stopped = false;
  get("uidLocateForm").listeners.submit({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true); assert.match(get("uidLocateStatus").textContent, /没有场景帧/);
  get("uidLocateForm").listeners.keydown({ key: "Enter", stopPropagation() { stopped = true; } });
  assert.equal(stopped, true);
  const html = readFileSync(new URL("../terminal_marl/visualization/web/index.html", import.meta.url), "utf8");
  assert.match(html, /<form id="uidLocateForm"[\s\S]*?id="uidLocateInput"[\s\S]*?<button[^>]*type="submit"[^>]*>定位<\/button>[\s\S]*?<\/form>/);
  assert.match(html, /for="uidLocateInput">定位无人机 UID/);
  assert.match(html, /id="uidLocateStatus"[^>]*aria-live="polite"/);
});

test("flight, event and details labels use UID including zero, missing and absent records", () => {
  const { h, get } = uidUiHarness();
  const uav = { uid: 0, aircraft_id: "original-aircraft-99", phase: "HOLD", soc: .5 };
  const event = { ...uav, outcome: "failed", time: 2 };
  const f = frame(2, { uavs: [{ uid: 12, aircraft_id: "AAA" }, uav, { aircraft_id: "missing-uid" }], terminal_events: [event, { aircraft_id: "also-missing-uid" }] });
  h.state.frames = [f]; h.state.selectedUid = 0;
  h.state.reference = { frames: [frame(2, { uavs: [{ ...uav, uid: 81 }] })] };
  h.renderFlights(f);
  const flightBody = get("flightTable").querySelector("tbody");
  assert.deepEqual(flightBody.children.map((row) => row.children[0].textContent), ["UID —", "UID 0", "UID 12"]);
  get("flightSearch").value = "original-aircraft"; h.renderFlights(f); assert.equal(flightBody.children.length, 0);
  get("flightSearch").value = "UID 0"; h.renderFlights(f); assert.equal(flightBody.children[0].children[0].textContent, "UID 0");
  h.renderEvents(f);
  assert.deepEqual(get("eventsTable").querySelector("tbody").children.map((row) => row.children[1].textContent), ["UID —", "UID 0"]);
  h.renderDetails(f); assert.equal(get("selectedTitle").textContent, "UID 0");
  assert.match(get("selectedState").textContent, /参考 UID 81 已匹配/);
  h.state.selectedEvent = event; h.renderDetails(frame(3)); assert.equal(get("selectedTitle").textContent, "UID 0");
  h.state.selectedEvent = null; h.renderDetails(frame(3)); assert.equal(get("selectedTitle").textContent, "UID 0 · 当前帧无数据");
  h.state.selectedUid = null; h.state.selectedEvent = { aircraft_id: "missing" };
  h.renderDetails(frame(3)); assert.equal(get("selectedTitle").textContent, "UID —");
});

test("resource owners display recorded UIDs while selected reservation filtering keeps raw IDs", () => {
  const { h, get } = uidUiHarness();
  const reservation = (owner_id, extra = {}) => ({ owner_id, commitment: "COMMITTED", interval_start: 1, interval_end: 2, ...extra });
  const f = frame(1, { uavs: [{ uid: 0, aircraft_id: "original-zero" }], terminal_events: [{ uid: 9, aircraft_id: "departed" }], rotating_slots: {
    pads: [{ id: 1, occupant_id: "original-zero", occupied_by: 0, reservations: [reservation("authoritative", { owner_uid: 4 }), reservation("departed"), reservation("unknown-123")] }],
    slots: [{ id: 7, reservations: [reservation("future", { future_interval: true, owner_uid: 8 }), reservation("original-zero"), reservation("UID 0"), reservation("other")] }],
  } });
  h.state.selectedUid = 0; h.renderResources(f);
  const lines = get("resourceContent").children.map((row) => row.textContent);
  assert.match(lines[0], /实际服务 UID 0/);
  assert.match(lines[1], /UID 4/); assert.match(lines[2], /UID 9/); assert.match(lines[3], /UID —/);
  const slots = lines.filter((line) => line.startsWith("SLOT"));
  assert.equal(slots.length, 2); assert.match(slots[0], /UID 0/); assert.match(slots[1], /UID 8/);
  assert.ok(lines.every((line) => !/original-zero|authoritative|departed|unknown-123/.test(line)));
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
