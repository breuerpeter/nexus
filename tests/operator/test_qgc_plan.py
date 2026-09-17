"""The QGroundControl ``.plan`` reader: JSON in, ``MISSION_ITEM_INT`` fields out.

No pymavlink here on purpose: the reader is pure JSON handling, so it tests without a MAVLink
stack and the command/frame numbers below are the wire constants written out.
"""

import json
import math
import pathlib

import pytest

import nexus
from nexus._src.operator.qgc_plan import read_plan

BOX_PLAN = pathlib.Path(nexus.__file__).parent / "examples" / "controllers" / "px4" / "box.plan"


def _item(command, *, lat=None, lon=None, alt=0.0, params=None, frame=3, auto=True):
    """One `.plan` SimpleItem, in the file's own shape, where params 5-7 are lat/lon/alt."""
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": alt,
        "AltitudeMode": 1,
        "autoContinue": auto,
        "command": command,
        "doJumpId": 1,
        "frame": frame,
        "params": params if params is not None else [0, 0, 0, None, lat, lon, alt],
        "type": "SimpleItem",
    }


def _plan(tmp_path, items, home=(47.5566, -122.1643, 0.0), **overrides):
    """Write a `.plan` carrying `items` and return its path."""
    doc = {
        "fileType": "Plan",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 15,
            "firmwareType": 12,
            "hoverSpeed": 5,
            "items": items,
            "plannedHomePosition": list(home),
            "vehicleType": 2,
            "version": 2,
        },
        "rallyPoints": {"points": [], "version": 2},
        "version": 1,
    }
    doc.update(overrides)
    path = tmp_path / "test.plan"
    path.write_text(json.dumps(doc))
    return path


def test_items_take_seq_from_file_order(tmp_path):
    # No implicit home item: the first real item is seq 0, which is what PX4 accepts.
    path = _plan(tmp_path, [_item(22, lat=47.0, lon=-122.0, alt=40.0), _item(16, lat=47.001, lon=-122.0, alt=40.0)])
    plan = read_plan(path)
    assert [i.seq for i in plan.items] == [0, 1]
    assert [i.command for i in plan.items] == [22, 16]


def test_lat_lon_become_dege7_integers(tmp_path):
    path = _plan(tmp_path, [_item(16, lat=47.5575009, lon=-122.1629651, alt=40.0)])
    (wp,) = read_plan(path).items
    assert wp.x == 475575009
    assert wp.y == -1221629651
    assert isinstance(wp.x, int) and isinstance(wp.y, int)
    assert wp.z == 40.0
    assert wp.frame == 3


def test_null_param_is_nan(tmp_path):
    # param4 null = "PX4 chooses the yaw"; NaN is how MAVLink spells no-value.
    path = _plan(tmp_path, [_item(16, lat=47.0, lon=-122.0, alt=40.0)])
    (wp,) = read_plan(path).items
    assert math.isnan(wp.params[3])
    assert wp.params[:3] == (0.0, 0.0, 0.0)


def test_positionless_item_reads_as_zero(tmp_path):
    # NAV_RETURN_TO_LAUNCH carries no fix; the file stores zeros/nulls there.
    path = _plan(tmp_path, [_item(20, params=[0, 0, 0, 0, None, None, None])])
    (rtl,) = read_plan(path).items
    assert (rtl.x, rtl.y, rtl.z) == (0, 0, 0.0)


def test_autocontinue_is_read(tmp_path):
    path = _plan(tmp_path, [_item(16, lat=47.0, lon=-122.0, auto=False)])
    assert read_plan(path).items[0].autocontinue is False


def test_home_position_is_read(tmp_path):
    path = _plan(tmp_path, [_item(16, lat=47.0, lon=-122.0)], home=(47.5566, -122.1643, 12.5))
    assert read_plan(path).home == (47.5566, -122.1643, 12.5)


def test_wrong_file_type_raises(tmp_path):
    path = _plan(tmp_path, [_item(16, lat=47.0, lon=-122.0)], fileType="Fence")
    with pytest.raises(ValueError, match=r"not a QGC \.plan"):
        read_plan(path)


def test_missing_mission_raises(tmp_path):
    path = tmp_path / "bare.plan"
    path.write_text(json.dumps({"fileType": "Plan", "version": 1}))
    with pytest.raises(ValueError, match="no 'mission' object"):
        read_plan(path)


def test_complex_item_raises(tmp_path):
    # A QGroundControl survey is a ComplexItem: it expands to many items in QGroundControl, and the
    # reader doesn't do that.
    survey = {"type": "ComplexItem", "complexItemType": "survey"}
    path = _plan(tmp_path, [survey])
    with pytest.raises(ValueError, match="only 'SimpleItem' is supported"):
        read_plan(path)


def test_short_params_raises(tmp_path):
    path = _plan(tmp_path, [_item(16, params=[0, 0, 0, 0, 47.0])])
    with pytest.raises(ValueError, match="has 5 params, expected 7"):
        read_plan(path)


# ---- the shipped box.plan, which px4_mission flies ----------------------------------------


def test_box_plan_is_takeoff_four_waypoints_and_rtl():
    plan = read_plan(BOX_PLAN)
    assert [i.command for i in plan.items] == [22, 16, 16, 16, 16, 20]
    assert [i.seq for i in plan.items] == [0, 1, 2, 3, 4, 5]
    assert plan.home == (47.5566, -122.1643, 0.0)  # the plan's home


def test_box_plan_sends_the_rtl_in_the_mission_frame():
    # Load-bearing, and a silent upload-killer: PX4 parses NAV_RETURN_TO_LAUNCH *only* in the
    # MAV_FRAME_MISSION branch, mavlink_mission.cpp:1602. Sent in a global frame it falls to
    # that switch's default at :1593 and PX4 refuses the *whole* mission with MAV_MISSION_UNSUPPORTED.
    plan = read_plan(BOX_PLAN)
    assert [i.frame for i in plan.items] == [3, 3, 3, 3, 3, 2]


def test_box_plan_flies_a_100m_box_at_40m():
    # The fixes, as the degE7 integers that go on the wire. Authored with the
    # Global Positioning System (GPS) kernel's constants, 111000 m/deg lat and 111000*cos(lat) m/deg
    # lon, so the box lands where it should in the sim world: _R_EARTH would put every corner ~0.3% out.
    plan = read_plan(BOX_PLAN)
    assert [(i.x, i.y) for i in plan.items] == [
        (475566000, -1221643000),  # takeoff, at home
        (475575009, -1221643000),  # north 100
        (475575009, -1221629651),  # north 100, east 100
        (475566000, -1221629651),  # east 100
        (475566000, -1221643000),  # back to home
        (0, 0),  # return to launch carries no fix
    ]
    assert [i.z for i in plan.items] == [40.0, 40.0, 40.0, 40.0, 40.0, 0.0]


def test_box_plan_leaves_the_waypoint_yaw_to_px4():
    for item in read_plan(BOX_PLAN).items[:5]:
        assert math.isnan(item.params[3])  # param4 NaN = PX4 chooses the heading
