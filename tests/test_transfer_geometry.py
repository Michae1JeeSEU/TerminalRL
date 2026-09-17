import math
import numpy as np
import pytest
from terminal_marl.core.config import SimConfig
from terminal_marl.core.trajectory_generator import TrajectoryGenerator
from terminal_marl.core.trajectory_time_parameterization import GeometricPath


@pytest.mark.parametrize('angle', [0., .05, .35, .5, 1., 1.9, 2., 3., 4., 5., 6.2])
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
    delta = np.diff(xy, axis=0)
    fraction = np.clip(-np.sum(xy[:-1]*delta, axis=1) / np.sum(delta*delta, axis=1), 0., 1.)
    closest = xy[:-1] + fraction[:, None]*delta
    assert np.linalg.norm(closest, axis=1).min() >= r1 - .05
    np.testing.assert_array_equal(path.points[[0, -1]], [p0, p3])
    if not repaired:
        control = min(.42 * np.linalg.norm(p3 - p0), 160.)
        old = GeometricPath.cubic_bezier(p0, p0 + control*t0, p3-control*t1, p3,
                                         samples=g.config.trajectory_bezier_samples)
        np.testing.assert_array_equal(path.points, old.points)
    else:
        for delta, tangent in ((path.points[1]-p0, t0), (p3-path.points[-2], t1)):
            assert np.dot(delta / np.linalg.norm(delta), tangent) > .995

def test_seed757006_legacy_nodes_hide_inward_chord():
    g=TrajectoryGenerator(SimConfig())
    p0=np.array([-330.592217743971,-225.186113175576,70.])
    p3=np.array([159.704874535755,-120.392495818961,40.])
    def tangent(p):
        angle=math.atan2(p[1],p[0]);return np.array([-math.sin(angle),math.cos(angle),0.])
    t0,t1=tangent(p0),tangent(p3)
    control=min(.42*np.linalg.norm(p3-p0),160.)
    old=GeometricPath.cubic_bezier(p0,p0+control*t0,p3-control*t1,p3,samples=41)
    def min_segment(path):
        xy=path.points[:,:2];delta=np.diff(xy,axis=0)
        f=np.clip(-np.sum(xy[:-1]*delta,axis=1)/np.sum(delta*delta,axis=1),0.,1.)
        return np.linalg.norm(xy[:-1]+f[:,None]*delta,axis=1).min()
    assert np.linalg.norm(old.points[:,:2],axis=1).min()>=200.-1e-4
    assert min_segment(old)<199.95
    fixed,repaired=g._transfer_path(p0,p3,t0,t1)
    assert repaired and min_segment(fixed)>=199.95
