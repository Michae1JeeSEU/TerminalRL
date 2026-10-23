"""Read-only identities for the existing Stage2 nodes and trajectory chains."""
import hashlib
import json


def stable_id(kind, value):
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"), allow_nan=False)
    return kind + ":" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def snapshot_identity(snapshot):
    return stable_id("snapshot", (snapshot.config.config_hash, snapshot.version, snapshot.time))


def node_identity(snapshot, aircraft_id, node):
    return stable_id("node", (snapshot_identity(snapshot), aircraft_id, node.ring, node.slot,
                              node.time, node.soc, node.peak, node.version))


def prefix_identity(snapshot, aircraft_id, plans):
    return stable_id("prefix", (snapshot_identity(snapshot), aircraft_id,
                                tuple(p.trajectory_hash for p in plans)))
