import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../terminal_marl/visualization/web/scene.js", import.meta.url), "utf8")
  .replace("export default class Scene", "class Scene");
const context = vm.createContext({ performance: { now: () => 0 } });
vm.runInContext(`${source}\nglobalThis.Scene = Scene;`, context);
const Scene = context.Scene;
const HistoryMap = vm.runInContext("Map", context);
const flying = { uid: 1, aircraft_id: "A1", phase: "HOLD_INNER", position: [60, 70, 20], soc: .3 };

function scene() {
  const calls = [];
  const stack = [];
  const ctx = new Proxy({
    globalAlpha: 1,
    save() { stack.push({ globalAlpha: this.globalAlpha, lineWidth: this.lineWidth, strokeStyle: this.strokeStyle }); },
    restore() { Object.assign(this, stack.pop()); },
    measureText: (text) => ({ width: text.length * 6 }),
    fillText(...args) { calls.push({ kind: "text", args, alpha: this.globalAlpha }); },
    fillRect: (...args) => calls.push({ kind: "rect", args }),
    arc: (...args) => calls.push({ kind: "arc", args }),
    stroke() { calls.push({ kind: "stroke", width: this.lineWidth, color: this.strokeStyle, alpha: this.globalAlpha }); },
  }, { get: (target, key) => key in target ? target[key] : () => {} });
  const instance = Object.assign(Object.create(Scene.prototype), {
    ctx, calls, frame: {}, metrics: { cssWidth: 900, cssHeight: 650, scale: 1 },
    layers: { labels: false, landingZones: true }, selectedUid: null,
    palette: ["#2864b7"], camera: { yaw: 0 }, _project: (p) => ({ x: p[0], y: p[1], depth: p[2] }),
  });
  return instance;
}

test("low SOC uses explicit adapter and normalized alerts with rotating-only legacy fallback", () => {
  assert.equal(Scene.energyCue({ ...flying, low_soc: true }).kind, "low_soc");
  for (const energy_alert of ["low_soc", "low soc", " LOW SOC "]) {
    assert.equal(Scene.energyCue({ ...flying, energy_alert }).label, "LOW SOC 30%");
  }
  assert.equal(Scene.energyCue(flying), null);
  assert.equal(Scene.energyCue({ ...flying, soc: .35 }, { console_replay_kind: "rotating_slots" }).kind, "low_soc");
  assert.equal(Scene.energyCue({ ...flying, soc: .351 }, { console_replay_kind: "rotating_slots" }), null);
  assert.equal(Scene.energyCue({ ...flying, low_soc: false, energy_alert: "low_soc" }, { console_replay_kind: "rotating_slots" }), null);
});

test("emergency declaration takes priority without inferring it from very low SOC", () => {
  for (const declaration of [{ emergency_landing: true }, { energy_alert: "emergency" }, { phase: "FINAL_EMERGENCY" }, { rescue_status: "ACTIVE" }]) {
    assert.equal(Scene.energyCue({ ...flying, low_soc: true, ...declaration }).label, "EMERGENCY 30%");
  }
  assert.equal(Scene.energyCue({ ...flying, soc: .001 }, { console_replay_kind: "rotating_slots" }).kind, "low_soc");
  for (const soc of [undefined, null, "0.3", NaN]) {
    assert.equal(Scene.energyCue({ ...flying, soc, low_soc: true }).label, "LOW SOC —");
    assert.equal(Scene.energyCue({ ...flying, soc }, { console_replay_kind: "rotating_slots" }), null);
  }
});

test("ground and terminal states suppress retained active declarations", () => {
  for (const status of ["LANDED", "LANDED_SERVICE", "SERVICE", "CLEARED", "COMPLETED", "ENERGY_FAILURE", "ENERGY_MISSION_FAILED", "CENSORED_DUE_TO_FAILURE", "REROUTED", "UNRELEASED_AT_FAILURE", "GROUND", "EXECUTED_HARD_SEPARATION_VIOLATION"]) {
    for (const key of ["phase", "status", "original_status"]) {
      const uav = { ...flying, phase: "FINAL_EMERGENCY", emergency_landing: true, low_soc: true, [key]: status };
      assert.equal(Scene.energyCue(uav), null, `${key}=${status}`);
      assert.equal(Scene.emergencyFlight(uav), false, `${key}=${status}`);
    }
  }
});

test("declared wait and reserved plans keep ordinary paths until the maneuver executes", () => {
  for (const rescue_status of ["WAIT", "RESERVED"]) {
    assert.equal(Scene.emergencyFlight({ ...flying, emergency_landing: true, rescue_status, committed: true, trajectory_kind: "FINAL_EMERGENCY" }), false);
    assert.equal(Scene.emergencyFlight({ ...flying, phase: "FINAL_NORMAL", emergency_landing: true, rescue_status, committed: true, trajectory_kind: "FINAL_NORMAL" }), false);
  }
  assert.equal(Scene.emergencyFlight({ ...flying, rescue_status: "ACTIVE" }), false);
  assert.equal(Scene.emergencyFlight({ ...flying, phase: "FINAL_EMERGENCY" }), true);
  assert.equal(Scene.emergencyFlight({ ...flying, phase: "VERTICAL_DESCENT", rescue_status: "ACTIVE" }), true);
  assert.equal(Scene.emergencyFlight({ ...flying, phase: "VERTICAL_DESCENT", trajectory_kind: "FINAL_EMERGENCY", committed: true }), true);
  assert.equal(Scene.emergencyFlight({ ...flying, phase: "VERTICAL_DESCENT", trajectory_kind: "FINAL_EMERGENCY", committed: false }), false);
});

test("actual emergency planned, flown, and reference paths retain distinct stroke semantics", () => {
  const instance = scene();
  const paths = [];
  instance._path = (...args) => paths.push(args);
  instance._drawReferenceUav = () => {};
  const points = [[1, 2, 3], [4, 5, 6]];
  const uav = { ...flying, phase: "FINAL_EMERGENCY", committed: true, trajectory_points: points };
  instance._drawCommitted(uav);
  assert.equal(paths[0][0][0], points[0]);
  assert.equal(paths[0][1], "#b23c43");
  assert.equal(paths[0][2], 4.5);
  assert.deepEqual(Array.from(paths[0][4]), [8, 4]);
  instance._drawTrail(uav, points);
  assert.equal(paths[1][0], points);
  assert.equal(paths[1][1], "#b23c43");
  assert.equal(paths[1][2], 3.5);
  assert.equal(paths[1][4], undefined);
  instance._drawReferenceFrame({ uavs: [uav] });
  assert.equal(paths[2][1], "#c98287");
  assert.deepEqual(Array.from(paths[2][4]), [2, 5]);
  instance._drawCommitted({ ...uav, phase: "HOLD_INNER", rescue_status: "WAIT", emergency_landing: true });
  assert.equal(paths[3][1], "#2864b76e");
  assert.deepEqual(Array.from(paths[3][4]), [7, 4]);
  for (const phase of ["HOLD_INNER", "TRANSFER_M_I", "LANDED_SERVICE"]) {
    instance._drawTrail({ ...uav, phase, trajectory_kind: "FINAL_EMERGENCY", emergency_landing: true }, points);
    assert.equal(paths.at(-1)[1], "#2864b788", `recorded ${phase} trail`);
  }
  instance._drawTrail({ ...uav, phase: "VERTICAL_DESCENT", rescue_status: "ACTIVE" }, points);
  assert.equal(paths.at(-1)[1], "#b23c43");
});

test("all airborne alerts draw readable tags without selection or ordinary labels", () => {
  const instance = scene();
  instance.frame = { landing_zones: [{ lz_id: 1, position: [250, 250, 0] }] };
  const uavs = Array.from({ length: 14 }, (_, uid) => ({ ...flying, uid, aircraft_id: `A${uid}`, position: [160 + uid % 4 * 170, 120 + Math.floor(uid / 4) * 110, 20], low_soc: true, emergency_landing: uid === 13 }));
  instance._drawEnergyAlerts([...uavs, { ...flying, uid: 99, phase: "ENERGY_FAILURE", emergency_landing: true }]);
  const labels = instance.calls.filter((call) => call.kind === "text");
  assert.equal(labels.length, 14);
  assert.match(labels[13].args[0], /^UID 13 · EMERGENCY 30%$/);
  assert.match(labels[0].args[0], /^UID 0 · LOW SOC 30%$/);
  assert.equal(labels.filter((call) => call.args[0].includes("LOW SOC")).length, 13);
  for (const call of instance.calls.filter((entry) => entry.kind === "rect")) {
    const [x, y, w, h] = call.args;
    assert.ok(x >= 4 && y >= 4 && x + w <= 896 && y + h <= 646);
    assert.equal(h, 19);
    // P1 label occupies x=[230,270], y=[256,274].
    assert.ok(x + w <= 230 || x >= 270 || y + h <= 256 || y >= 274);
  }
});

test("render schedules safety overlay after every aircraft even with 14 UAVs and labels off", () => {
  const instance = scene();
  const order = [];
  instance.frame = { uavs: Array.from({ length: 14 }, (_, uid) => ({ ...flying, uid, low_soc: true })) };
  instance.layers = { labels: false };
  instance.camera = { zoom: 1 };
  instance._resize = () => ({ width: 900, height: 650 });
  instance._bounds = () => ({ extent: 100 });
  instance._depth = () => 0;
  instance._drawReferenceFrame = () => {};
  instance._drawTerminalEvents = () => {};
  instance._drawUav = (_uav, _conflict, showLabel) => { assert.equal(showLabel, false); order.push("uav"); };
  instance._drawEnergyAlerts = (uavs) => { assert.equal(uavs.length, 14); order.push("alerts"); };
  instance.render();
  assert.equal(order.filter((value) => value === "uav").length, 14);
  assert.equal(order.at(-1), "alerts");
});

test("alert UIDs remain complete and never fall back to aircraft IDs", () => {
  const instance = scene();
  instance._drawEnergyAlerts([
    { ...flying, uid: "12345678901234567890", aircraft_id: "initial_middle_03", low_soc: true },
    { ...flying, uid: null, aircraft_id: "initial_middle_04", low_soc: true },
  ]);
  const labels = instance.calls.filter((call) => call.kind === "text").map((call) => call.args[0]);
  assert.equal(labels.length, 2);
  assert.notEqual(labels[0], labels[1]);
  assert.equal(labels[0], "UID 12345678901234567890 · LOW SOC 30%");
  assert.equal(labels[1], "UID — · LOW SOC 30%");
});

test("ordinary, selected, hovered, reference and failed labels use explicit UIDs", () => {
  const instance = scene(); instance.layers.labels = true;
  const uav = { ...flying, uid: 0, aircraft_id: "original-99" };
  instance.frame = { uavs: [uav] }; instance.referenceFrame = { uavs: [uav] };
  instance._drawUav(uav, false, true);
  instance.selectedUid = 0; instance._drawUav(uav, false, false);
  instance.selectedUid = null; instance.hoverUid = 0; instance._drawUav(uav, false, false);
  instance._drawReferenceUav(uav);
  instance._drawUav({ ...uav, status: "ENERGY_FAILURE" }, false, false);
  instance._drawTerminalEvents({ terminal_events: [{ ...uav, outcome: "failed" }] });
  instance._drawUav({ ...uav, uid: null }, false, true);
  const labels = instance.calls.filter((call) => call.kind === "text").map((call) => call.args[0]);
  assert.deepEqual(labels, ["UID 0", "UID 0", "UID 0", "R UID 0", "UID 0 · ENERGY_FAILURE", "UID 0 · failed", "UID —"]);
});

test("UID zero focus and highlight exclude missing UID aircraft", () => {
  const instance = scene();
  const missing = { ...flying, uid: null, position: [100, 110, 20] };
  const zero = { ...flying, uid: 0, position: [20, 30, 40] };
  instance.frame = { uavs: [missing, zero] };
  instance.selectedUid = 0; instance.hoverUid = 0;
  instance._drawUav(missing, false, false);
  assert.equal(instance.calls.some((call) => call.kind === "text" || call.kind === "stroke" && call.color === "#202428"), false);
  instance.render = () => {};
  instance.camera.panX = 0; instance.camera.panY = 0;
  assert.equal(instance.focusOn(0, instance.frame), true);
  assert.equal(instance.camera.panX, 430); assert.equal(instance.camera.panY, 295);
  assert.equal(instance.focusOn(null, instance.frame), false);
});

test("all UID toggle labels every aircraft once above the density limit with ordinary labels off", () => {
  const instance = scene();
  const uavs = Array.from({ length: 12 }, (_, uid) => ({ ...flying, uid, position: [50 + uid * 15, 70, 20], ...(uid === 11 ? { status: "ENERGY_FAILURE" } : {}) }));
  instance.frame = { uavs }; instance.layers = { labels: false, allUids: false };
  instance.camera.zoom = 1; instance._resize = () => ({ width: 900, height: 650 });
  instance._bounds = () => ({ extent: 100 }); instance._depth = () => 0;
  instance._drawReferenceFrame = () => {}; instance._drawTerminalEvents = () => {}; instance._drawEnergyAlerts = () => {};
  instance.render(); assert.equal(instance.calls.filter((call) => call.kind === "text").length, 0);
  instance.calls.length = 0; instance.selectedUid = 0;
  instance.setLayers({ allUids: true });
  const labels = instance.calls.filter((call) => call.kind === "text").map((call) => call.args[0]);
  assert.equal(labels.length, 12); assert.equal(labels[0], "UID 0"); assert.equal(labels[10], "UID 10");
  assert.equal(labels[11], "UID 11 · ENERGY_FAILURE");
  instance.calls.length = 0; instance.selectedUid = null; instance.setLayers({ allUids: false });
  assert.equal(instance.calls.filter((call) => call.kind === "text").length, 0);
  instance.setLayers({ allUids: true }); instance.calls.length = 0;
  instance.referenceFrame = { uavs }; instance._drawReferenceUav(uavs[5]);
  assert.equal(instance.calls.filter((call) => call.kind === "text")[0].args[0], "R UID 5");
});

test("low SOC and emergency bodies and safety outlines have visibly bolder strokes", () => {
  const instance = scene();
  for (const [declaration, width] of [[{}, 1.5], [{ low_soc: true }, 2.8], [{ emergency_landing: true }, 3.4]]) {
    instance.calls.length = 0;
    instance._drawUav({ ...flying, ...declaration }, false, false);
    const body = instance.calls.filter((call) => call.kind === "stroke" && call.color === "#2864b7");
    assert.equal(body.length, 5);
    assert.ok(body.every((call) => call.width === width));
  }
  instance.calls.length = 0;
  instance._drawEnergyAlerts([{ ...flying, low_soc: true }], 0);
  assert.ok(instance.calls.some((call) => call.kind === "stroke" && call.color === "#a66b13" && call.width === 3));
  instance.calls.length = 0;
  instance._drawEnergyAlerts([{ ...flying, emergency_landing: true }], 0);
  assert.equal(instance.calls.filter((call) => call.kind === "stroke" && call.color === "#b23c43" && call.width === 3.5).length, 2);
});

test("emergency outer halo pulses on wall time while paused aircraft and tags remain visible", () => {
  const instance = scene();
  instance.frame = { time: 123 };
  const uav = { ...flying, emergency_landing: true };
  const phases = [0, 500].map((wallTime) => {
    instance.calls.length = 0;
    instance._drawUav(uav, false, false);
    instance._drawEnergyAlerts([uav], wallTime);
    return instance.calls.slice();
  });
  const halos = phases.map((calls) => calls.find((call) => call.kind === "stroke" && call.color === "#b23c43" && call.width === 2.5));
  assert.ok(halos[0].alpha > halos[1].alpha);
  assert.ok(halos[1].alpha > 0);
  for (const calls of phases) {
    assert.equal(calls.filter((call) => call.kind === "stroke" && call.color === "#2864b7" && call.width === 3.4).length, 5);
    const tag = calls.filter((call) => call.kind === "text");
    assert.equal(tag.length, 1);
    assert.match(tag[0].args[0], /EMERGENCY 30%/);
    assert.equal(tag[0].alpha, 1);
    assert.equal(calls.filter((call) => call.kind === "stroke" && call.color === "#b23c43" && call.width === 3.5 && call.alpha === 1).length, 2);
  }
  assert.equal(instance.frame.time, 123);
});

function renderHistory(instance, uavs, trails = true) {
  instance.frame = { uavs };
  instance.layers = { trails, labels: false };
  instance.camera = { zoom: 1 };
  instance._resize = () => ({ width: 900, height: 650 });
  instance._bounds = () => ({ extent: 100 });
  instance._depth = () => 0;
  instance._drawReferenceFrame = () => {};
  instance._drawTerminalEvents = () => {};
  instance._drawUav = () => {};
  instance._drawEnergyAlerts = () => {};
  const paths = [];
  instance._path = (...args) => paths.push(args);
  instance.render();
  return paths;
}

test("dedicated emergency history never joins null or invalid point gaps", () => {
  const instance = scene();
  const a = [[1, 2, 20], [2, 3, 15]];
  const b = [[3, 4, 10], [3, 4, 0]];
  instance.history = new HistoryMap([
    ["1:emergency", [...a, null, ...b, [NaN, 2, 3], [9, 9, 0]]],
    ["2:HOLD", [[40, 30, 20], [41, 31, 20]]],
  ]);
  const paths = renderHistory(instance, []);
  assert.equal(paths.length, 2);
  assert.deepEqual(Array.from(paths[0][0]), a);
  assert.deepEqual(Array.from(paths[1][0]), b);
  for (const [, color, width, , dash] of paths) {
    assert.equal(color, "#b23c43");
    assert.equal(width, 3.5);
    assert.equal(dash, undefined);
  }
});

test("flown emergency history persists after service, clearing, and departure with trails enabled", () => {
  const instance = scene();
  const points = [[10, 20, 15], [10, 20, 8], [10, 20, 0]];
  instance.history = new HistoryMap([["1:emergency", points]]);
  for (const uavs of [[{ ...flying, phase: "LANDED_SERVICE" }], [{ ...flying, phase: "CLEARED" }], []]) {
    const paths = renderHistory(instance, uavs);
    assert.equal(paths.length, 1);
    assert.deepEqual(Array.from(paths[0][0]), points);
    assert.equal(paths[0][1], "#b23c43");
  }
  assert.equal(renderHistory(instance, [], false).length, 0);
});

test("dedicated rescue history replaces its phase duplicate while legacy phase histories still draw", () => {
  const instance = scene();
  const points = [[10, 20, 15], [10, 20, 8]];
  const uav = { ...flying, phase: "FINAL_EMERGENCY" };
  instance.history = new HistoryMap([["1:FINAL_EMERGENCY", points], ["1:emergency", points]]);
  assert.equal(renderHistory(instance, [uav]).length, 1);
  instance.history.delete("1:emergency");
  const legacy = renderHistory(instance, [uav]);
  assert.equal(legacy.length, 1);
  assert.equal(legacy[0][1], "#b23c43");
  instance.history.set("1:emergency", points);
  instance.history.set("1:HOLD_INNER", points);
  const waiting = renderHistory(instance, [{ ...flying, phase: "HOLD_INNER", emergency_landing: true }]);
  assert.equal(waiting.length, 2);
  assert.equal(waiting[0][1], "#2864b788");
  assert.equal(waiting[1][1], "#b23c43");
});
