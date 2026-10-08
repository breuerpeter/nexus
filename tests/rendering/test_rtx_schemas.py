"""A vehicle declares each RTX sensor with one applied schema on its prim, and the Kit peer is a required peer.

Real builds on the Warp CPU backend of the local fixture vehicle in ``tests/usd/sensor_vehicle.py``,
with the peer mapping sending the Kit peer to its fake. The docker
daemon is the system boundary, and the stand-in daemon keeps any container a run would start from
reaching a real one. Skipped without newton or pxr.
"""

import logging

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from pxr import Usd

from nexus_sim._src.core.registry import default_registry
from nexus_sim._src.peers.kit.fake import KitFake
from nexus_sim._src.peers.px4_sitl.fake import Px4Fake
from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu", "daemon")

CAMERA = sv.prim(
    "Cam", "NexusCameraAPI", "int nexus:width = 640\nint nexus:height = 480\nfloat nexus:rate = 30", kind="Camera"
)
THERMAL = sv.prim(
    "Ir", "NexusThermalCameraAPI", "int nexus:width = 320\nint nexus:height = 240\nfloat nexus:rate = 10", kind="Camera"
)
LIDAR = sv.prim("Lidar", "NexusLidarAPI", "float nexus:rate = 10", kind="OmniLidar")


def _kit() -> tuple[type, list]:
    """A fake Kit peer class for the peer mapping, and the list of every peer of it a run starts."""
    started = []

    class Kit(KitFake):
        def start(self) -> None:
            started.append(self)
            super().start()

    return Kit, started


def test_a_camera_a_thermal_camera_and_a_lidar_schema_each_build_their_sensor_with_the_authored_resolution_and_rate(
    tmp_path,
):
    """A camera schema or a thermal camera schema on a `Camera` prim, and a lidar schema on an `OmniLidar` prim, builds that sensor with its authored resolution and rate.

    Given a fixture with one prim of each and the peer mapping sending Kit to its fake, when built, then
    the loop holds one sensor per prim, of the class the default registry maps its schema to, with the
    authored width, height and rate.
    """
    kit, _ = _kit()
    loop = sv.build(sv.vehicle(tmp_path, CAMERA + THERMAL + LIDAR), peers={"kit": kit})
    built = [(type(s), s.rate) for s in loop.sensors]
    sizes = [(s.width, s.height) for s in loop.sensors[:2]]  # the lidar declares no resolution
    loop.close()

    registry = default_registry()
    assert (built, sizes) == (
        [
            (registry.resolve("NexusCameraAPI"), 30.0),
            (registry.resolve("NexusThermalCameraAPI"), 10.0),
            (registry.resolve("NexusLidarAPI"), 10.0),
        ],
        [(640, 480), (320, 240)],
    )


def test_a_resolution_or_rate_a_camera_prim_does_not_author_takes_the_schemas_fallback_with_no_warning(
    tmp_path, caplog
):
    """A resolution or rate the prim doesn't author takes the schema's fallback, with no warning.

    Given a `Camera` prim applying the camera schema with nothing authored, when built, then the sensor
    carries the fallbacks the plugin defines and the log holds no warning that names the prim.
    """
    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NexusCameraAPI")
    fallbacks = [definition.GetAttributeFallbackValue(f"nexus:{name}") for name in ("width", "height", "rate")]
    kit, _ = _kit()

    with caplog.at_level(logging.WARNING, logger="nexus"):
        loop = sv.build(sv.vehicle(tmp_path, sv.prim("Cam", "NexusCameraAPI", kind="Camera")), peers={"kit": kit})
    carried = [[s.width, s.height, s.rate] for s in loop.sensors]
    loop.close()
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "Cam" in r.getMessage()]

    assert (carried, warnings) == ([fallbacks], [])


def test_an_rtx_sensor_rides_its_parent_body_found_by_prim_path(tmp_path):
    """An RTX sensor rides its parent body, found by prim path.

    Given a fixture with two bodies of one leaf name, `body_frd`, under different parents and a camera
    under the second, when built, then the camera rides the second body.
    """
    kit, _ = _kit()
    loop = sv.build(sv.vehicle(tmp_path, mast=sv.prim("Cam", "NexusCameraAPI", kind="Camera")), peers={"kit": kit})
    second = list(loop.physics.model.body_label).index(sv.MAST)
    rides = [s.body for s in loop.sensors]
    loop.close()

    assert (second, rides) == (1, [1])


def test_a_camera_schema_on_a_prim_of_the_wrong_type_fails_the_build_and_names_the_prim(tmp_path):
    """An RTX sensor schema on a prim of the wrong type fails the build and names the prim.

    Given the camera schema on an Xform, when built, then the build fails naming the prim path and the
    type the schema needs, `Camera`.
    """
    kit, _ = _kit()

    with pytest.raises(ValueError) as err:
        sv.build(sv.vehicle(tmp_path, sv.prim("NotACam", "NexusCameraAPI")), peers={"kit": kit}).close()

    assert [text in str(err.value) for text in (f"{sv.BODY}/NotACam", "Camera")] == [True, True], str(err.value)


def test_a_camera_prim_that_applies_no_sensor_schema_is_not_a_sensor_and_starts_no_peer(tmp_path):
    """A `Camera` or `OmniLidar` prim that applies no sensor schema isn't a sensor and starts no peer.

    Given a fixture whose only `Camera` prim applies no schema, when built, then the loop holds no
    sensor and the Kit peer never starts.
    """
    kit, started = _kit()

    loop = sv.build(sv.vehicle(tmp_path, sv.prim("Bare", None, kind="Camera")), peers={"kit": kit})
    sensors = list(loop.sensors)
    loop.close()

    assert (sensors, started) == ([], [])


def test_a_vehicle_with_rtx_sensors_starts_the_kit_peer_once_and_stops_it_with_the_run(tmp_path):
    """A vehicle with RTX sensors starts the Kit peer once, as a required peer, and stops it with the run.

    Given a fixture with two cameras and a lidar, none of which names a peer, and the peer mapping
    sending Kit to its fake, when the run builds, steps and closes, then one fake started, and that one
    is down.
    """
    kit, started = _kit()
    second = sv.prim("Cam2", "NexusCameraAPI", kind="Camera")

    loop = sv.build(sv.vehicle(tmp_path, CAMERA + second + LIDAR), peers={"kit": kit})
    sv.steps(loop, 2)

    assert [peer.alive() for peer in started] == [False]


def test_a_vehicle_with_no_rtx_sensor_schema_starts_no_kit_peer(tmp_path):
    """A vehicle with no RTX sensor schema starts no Kit peer.

    Given a fixture that declares analytic sensors only, an Inertial Measurement Unit (IMU), a
    magnetometer, a barometer and a Global Positioning System (GPS) receiver, when built with the PX4
    peer and the Kit peer each sent to its fake, then the Kit peer never starts.
    """
    analytic = "".join(
        sv.prim(name, schema)
        for name, schema in (
            ("Imu", "NexusImuAPI"),
            ("Mag", "NexusMagAPI"),
            ("Baro", "NexusBaroAPI"),
            ("Gps", "NexusGpsAPI"),
        )
    )
    kit, started = _kit()

    path = sv.vehicle(tmp_path, analytic, px4=True)
    sv.build(path, components=default_registry(), peers={"px4_sitl": Px4Fake, "kit": kit}).close()

    assert started == []


def test_an_imu_and_a_camera_each_receive_the_rate_they_author(tmp_path):
    """Every sensor schema shares one rate attribute, and each sensor receives its authored rate.

    Given a fixture that authors the rate on an Inertial Measurement Unit (IMU), 50 Hz, and on a camera, 30 Hz, when built, then each
    sensor carries its rate.
    """
    kit, _ = _kit()
    prims = sv.prim("Imu0", "NexusImuAPI", "float nexus:rate = 50") + sv.prim(
        "Cam", "NexusCameraAPI", "float nexus:rate = 30", kind="Camera"
    )

    loop = sv.build(sv.vehicle(tmp_path, prims), peers={"kit": kit})
    rates = [s.rate for s in loop.sensors]
    loop.close()

    assert rates == [50.0, 30.0]
