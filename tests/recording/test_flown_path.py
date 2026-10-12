"""The flown path: a run's recording carries the trail beside the airframe's recorded series.

Real end-to-end on the Warp CPU backend: a one-body Universal Scene Description (USD) vehicle in
free fall, the loop driving a short run with its Recorder, then the ``.rrd`` it wrote. Skipped if rerun, newton or pxr are missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")
pytest.importorskip("pxr")

from nexus_uav._src.core.interfaces import Component, Stage

DT = 0.004  # 250 Hz control ticks
STEPS = 30  # ticks, well past the trail's two-point floor


def _author_min_usd(path: str) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Cube.Define(stage, "/Vehicle/body")
    body.GetSizeAttr().Set(0.2)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim()).GetMassAttr().Set(1.5)
    stage.GetRootLayer().Save()


class _Actuator(Component):
    """Writes no forces: the body flies the trail under gravity alone."""

    def stages(self):
        return [Stage("forces", "device", lambda tick: None)]


class _Controller(Component):
    """One device stage that commands nothing, so the body flies under gravity alone."""

    def connect(self, links) -> None:
        pass

    def stages(self):
        return [Stage("act", "device", lambda tick: None)]

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def flight_paths(tmp_path_factory, rrd_entities):
    """Fly a short run once and return the entity paths its recording carries."""
    import newton
    import warp as wp

    from nexus_uav._src.core.clock import Clock
    from nexus_uav._src.core.loop import Loop
    from nexus_uav._src.logging import Logger
    from nexus_uav._src.physics.physics import NewtonPhysics
    from nexus_uav._src.physics.vehicle import VehicleUsd
    from nexus_uav._src.recording import Recorder

    tmp = tmp_path_factory.mktemp("flight")
    usd = str(tmp / "mini.usda")
    _author_min_usd(usd)
    cfg = {"physics": {"dt": DT, "solver": "semi_implicit", "contacts": False, "spawn": {"pos": [0.0, 0.0, 5.0]}}}
    rrd = str(tmp / "flight.rrd")
    with wp.ScopedDevice("cpu"):
        mb = newton.ModelBuilder()
        VehicleUsd({"usd_path": usd}).build(mb)
        model = mb.finalize()
        orch = Loop(
            clock=Clock(DT),
            physics=NewtonPhysics(model=model, cfg=cfg),
            actuator=_Actuator(),
            sensors=[],
            controller=_Controller(),
            logger=Logger(model, serve=False, record_to_rrd=rrd),
            recorder=Recorder(),
            max_steps=STEPS,
        )
        orch.run()
    return rrd_entities(rrd)


def test_run_records_the_flown_path(flight_paths):
    """The trail the viewer draws beside the vehicle is in the recording."""
    assert "/sim/vehicle/trajectory" in flight_paths, flight_paths


def test_run_records_the_airframe_position_series(flight_paths):
    """The history that draws the flown path is in the recording too."""
    assert "/sim/vehicle/body/body/position" in flight_paths, flight_paths
