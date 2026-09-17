import math
import numpy as np
import pytest
from terminal_marl.core.config import SimConfig
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.trajectory_time_parameterization import GeometricPath


@pytest.mark.parametrize('angle', [0., .05, .5, 1., 2., 3., 4., 5., 6.2])
@pytest.mark.parametrize('radii', [(600., 400.), (400., 200.)])
def test_transfer_never_reverses_or_crosses_inner_disk(angle, radii):
    g = TrajectoryGenerator(SimConfig())
    r0, r1 = radii
    p0 = np.array([r0, 0., 70.])
    p3 = np.array([r1 * math.cos(angle), r1 * math.sin(angle), 40.])
    t0, t1 = np.array([0., 1., 0.]), np.array([-math.sin(angle), math.cos(angle), 0.])
    path, repaired = g._transfer_path(p0, p3, t0, t1)
    xy = path.points[:, :2]
    radius = np.linalg.norm(xy, axis=1)
    assert radius.min() >= r1 - 1e-4 and radius.max() <= r0 + 1e-4
    assert np.all(xy[:-1, 0] * xy[1:, 1] - xy[:-1, 1] * xy[1:, 0] >= -1e-8)
    # Execution interpolates chords between nodes, so check those too.
    mid = .5 * (xy[:-1] + xy[1:])
    assert np.linalg.norm(mid, axis=1).min() >= r1 - .05
    np.testing.assert_array_equal(path.points[[0, -1]], [p0, p3])
    if not repaired:
        control = min(.42 * np.linalg.norm(p3 - p0), 160.)
        old = GeometricPath.cubic_bezier(p0, p0 + control*t0, p3-control*t1, p3,
                                         samples=g.config.trajectory_bezier_samples)
        np.testing.assert_array_equal(path.points, old.points)
    else:
        for delta, tangent in ((path.points[1]-p0, t0), (p3-path.points[-2], t1)):
            assert np.dot(delta / np.linalg.norm(delta), tangent) > .995
