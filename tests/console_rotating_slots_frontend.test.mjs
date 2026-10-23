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
  vm.runInContext(`${source}\nglobalThis.harness = {state, sameAircraft, discreteReplay, interpolateFrame, displayFrame, renderFlights, appendServerPayload, metricsForFrame, referenceAtTime, validateReplayPayload, formatSoc, requestJson};`, context);
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
