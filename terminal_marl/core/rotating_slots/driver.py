"""Deterministic acceptance driver, deliberately not a research baseline."""
from .trajectory import generate_transfer


class DeterministicDriver:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, world):
        self.calls.append(snapshot.time)
        for request in snapshot.requests:
            if request.id not in world.aircraft:
                world.admit(request.id)
        for a in snapshot.aircraft:
            if (a.id in world.active or a.id in world.soft or a.emergency.declared or
                    a.emergency.eligible or a.emergency.pending):
                continue
            if a.status == "HOLD_INNER":
                world.try_final(a, emergency=False)
            elif a.status in ("HOLD_OUTER", "HOLD_MIDDLE"):
                count = 28 if a.ring == "outer" else 18
                for target in range(count):
                    result = generate_transfer(a.id, a.ring, a.slot, target, world.time,
                                               source_version=a.version, config=world.config)
                    if result.plan is None or result.status != "FEASIBLE":
                        world.reject(result.reason or result.status, a.id, target_slot=target, candidate=True)
                        continue
                    if world.submit((result.plan,)).accepted:
                        break


def run_deterministic(world, until=5400.):
    driver = DeterministicDriver()
    world.advance(until, driver)
    return world
