"""The flown path: a run's recording carries the trail beside the airframe's recorded series.

Real end-to-end on the Warp CPU backend: a one-body Universal Scene Description (USD) vehicle in
free fall, the orchestrator driving a short run with a Recorder attached, then the ``.rrd`` it
wrote. Skipped if rerun, newton or pxr are missing.
"""

import pytest

pytest.importorskip("rerun")
pytest.importorskip("newton")
pytest.importorskip("pxr")

DT = 0.004  # 250 Hz control ticks
STEPS = 30  # at the Logger's 50 Hz log rate that is 6 logged ticks, above the trail's two-point floor


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


class _Actuator:
    """Writes no forces: the body flies the trail under gravity alone."""

    capturable = False

    def forces(self, controls, state, env) -> None:
        pass


class _Controller:
    """Answers the pre-roll at once, then hands back empty controls every tick."""

    host_boundary = False
    capturable = False

    def connect(self) -> None:
        pass

    def exchange(self, meas, t, timeout=None) -> dict:
        return {}

    def close(self) -> None:
        pass


@pytest.fixture(scope="module")
def flight_paths(tmp_path_factory, rrd_entities):
    """Fly a short run once and return the entity paths its recording carries."""
    import newton
    import warp as wp

    from nexus._src.core.clock import Clock
    from nexus._src.core.environment import ConstantEnvironment
    from nexus._src.core.orchestrator import Orchestrator
    from nexus._src.logging import Logger
    from nexus._src.physics.builders.usd import USDBuilder
    from nexus._src.physics.physics import NewtonPhysics
    from nexus._src.recording import Recorder

    tmp = tmp_path_factory.mktemp("flight")
    usd = str(tmp / "mini.usda")
    _author_min_usd(usd)
    cfg = {"physics": {"dt": DT, "solver": "semi_implicit", "contacts": False, "spawn": {"pos": [0.0, 0.0, 5.0]}}}
    rrd = str(tmp / "flight.rrd")
    with wp.ScopedDevice("cpu"):
        mb = newton.ModelBuilder()
        USDBuilder({"usd_path": usd}, None).build(mb)
        model = mb.finalize()
        orch = Orchestrator(
            clock=Clock(DT),
            environment=ConstantEnvironment(),
            physics=NewtonPhysics(model=model, cfg=cfg),
            actuator=_Actuator(),
            sensors=[],
            controller=_Controller(),
            logger=Logger(model, serve=False, record_to_rrd=rrd),
            max_steps=STEPS,
        )
        orch.attach_recorder(Recorder(dt=DT))
        orch.run()
    return rrd_entities(rrd)


def test_run_records_the_flown_path(flight_paths):
    """The trail the viewer draws beside the vehicle is in the recording."""
    assert "/physics/trajectory" in flight_paths, flight_paths


def test_run_records_the_airframe_position_series(flight_paths):
    """The channel the flown path is drawn from is in the recording too."""
    assert any(p.startswith("/recording/physics/body/") and p.endswith("/position") for p in flight_paths), flight_paths
