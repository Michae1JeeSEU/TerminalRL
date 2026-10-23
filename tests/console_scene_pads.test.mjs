import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../terminal_marl/visualization/web/scene.js", import.meta.url), "utf8")
  .replace("export default class Scene", "class Scene");
const context = vm.createContext({});
vm.runInContext(`${source}\nglobalThis.Scene = Scene;`, context);

function padState(zone, pads = []) {
  return context.Scene.prototype._landingZoneState.call({ frame: { rotating_slots: { pads } } }, zone);
}

test("actual recorded pad occupancy takes precedence over reservation colors", () => {
  for (const status of ["free", "reserved", "soft_pending", "unknown"]) {
    assert.equal(padState({ lz_id: 0, status }, [{ id: 0, occupant_id: "A-0", state: "reserved" }]), "occupied");
  }
  assert.equal(padState({ lz_id: 0, status: "occupied" }), "occupied");
  assert.equal(padState({ lz_id: 0, status: "free" }, [{ id: 0, state: "occupied" }]), "occupied");
});

test("landing zone states include free, reserved and removable soft commitments", () => {
  assert.equal(padState({ lz_id: 0, status: "free" }), "free");
  assert.equal(padState({ lz_id: 1, status: "RESERVED" }), "reserved");
  assert.equal(padState({ lz_id: 2, status: "soft_pending" }), "soft");
  assert.equal(padState({ lz_id: 3, status: "soft" }), "soft");
});

test("recorded pad state fills unknown zone status without guessing from calendars", () => {
  assert.equal(padState({ lz_id: 2, status: "unknown" }, [{ id: "2", state: "soft_pending" }]), "soft");
  assert.equal(padState({ lz_id: 2, status: "unknown" }, [{ id: 2, state: "reserved", occupant_id: null,
    reservations: [{ future_interval: true, commitment: "COMMITTED" }] }]), "reserved");
  assert.equal(padState({ lz_id: 2, status: "unknown" }, [{ id: 2, reservations: [{ owner_id: "A-1" }] }]), "unknown");
  assert.equal(padState({ lz_id: 2, status: "unknown" }, [{ id: 3, state: "occupied", occupant_id: "A-1" }]), "unknown");
  assert.equal(padState({ lz_id: 2, status: "unrecognized" }), "unknown");
});

test("known zone state remains authoritative for reservation display", () => {
  assert.equal(padState({ lz_id: 1, status: "reserved" }, [{ id: 1, state: "soft_pending" }]), "reserved");
  assert.equal(padState({ lz_id: 1, status: "free" }, [{ id: 1, state: "reserved", occupant_id: null }]), "free");
});
