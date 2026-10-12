"""A sensor's prim is a site in Newton's model: the build loads it from the vehicle's Universal Scene
Description (USD) file, on the body it sits under, and a sensor schema on a prim that's no geometry prim
fails the build.

Real builds on the Warp CPU backend of the local fixture vehicle in ``tests/usd/sensor_vehicle.py``, with
the sensor prim a test adds under its base body. Skipped without newton or pxr.
"""

import pytest

pytest.importorskip("newton")
pytest.importorskip("pxr")

from newton import ShapeFlags

from tests.usd import sensor_vehicle as sv

pytestmark = pytest.mark.usefixtures("warp_cpu")

# An Inertial Measurement Unit (IMU) on a sphere 10 cm forward, 2 cm right and 3 cm up of its body's origin, in the body's Forward
# Right Down (FRD) axes, turned 90 degrees about the body's Z.
MOUNT = (0.1, 0.02, -0.03)
TURNED = (0.0, 0.0, 0.7071, 0.7071)  # the quaternion (x, y, z, w) of 90 degrees about Z, to four decimals
IMU_ON_SPHERE = sv.prim(
    "Imu0",
    "NexusImuAPI",
    "double radius = 0.01\n"
    f"double3 xformOp:translate = {MOUNT}\n"
    "float xformOp:rotateZ = 90\n"
    'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]',
    kind="Sphere",
)
IMU_ON_XFORM = sv.prim("Imu0", "NexusImuAPI", kind="Xform")


def _sites(model) -> dict:
    """Each site of `model` by its label: the label of the body it sits on, its translation and its rotation
    in the body's axes, to four decimals, the rotation's scalar part made positive.
    """
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    sites = {}
    for i in range(model.shape_count):
        if not flags[i] & ShapeFlags.SITE:
            continue
        p, q = transforms[i][:3], transforms[i][3:]
        if q[3] < 0.0:
            q = -q
        sites[str(model.shape_label[i])] = (
            str(model.body_label[bodies[i]]),
            tuple(round(float(x), 4) for x in p),
            tuple(round(float(x), 4) for x in q),
        )
    return sites


def test_a_sensors_prim_is_a_site_on_its_body_at_its_offset_and_rotation(tmp_path):
    """A sensor's prim with only its sensor schema authored is a site in the model: on the body it sits
    under, at the prim's offset and rotation, found by the prim's path.

    Given a vehicle whose IMU prim is a `Sphere` under a body, translated and rotated, when the run builds,
    then the model holds one site on that body with that transform under the prim's path.
    """
    loop = sv.build(sv.vehicle(tmp_path, IMU_ON_SPHERE))
    sites = _sites(loop.physics.model)
    loop.close()

    assert sites == {f"{sv.BODY}/Imu0": (sv.BODY, MOUNT, TURNED)}


def test_a_sensor_schema_on_a_prim_that_is_no_geometry_prim_fails_the_build_and_names_it(tmp_path):
    """A sensor schema on a prim that's no geometry prim fails the build and names the prim and its type.

    Given a vehicle whose IMU prim is an `Xform`, when the run builds, then it fails, and the message names
    the prim and `Xform`.
    """
    with pytest.raises(ValueError, match=rf"{sv.BODY}/Imu0.*Xform"):
        sv.build(sv.vehicle(tmp_path, IMU_ON_XFORM))
