"""The vehicle's motors: every ``NewtonActuator`` prim the vehicle's Universal Scene Description (USD) file
authors, which ``add_usd`` parses onto ``model.actuators``, stepped before each solver step.

A Newton motor reads its joint's state and the control's targets and writes the joint's torque into
``control.joint_f``, which :meth:`Motors.step` zeroes first, where Newton's docs place the call. A motor's
controller keeps state, so each motor holds two copies of it, and the step copies the next into the current
in place: a Python swap would freeze at the binding a CUDA graph captured.

Physics steps the motors of a run, and the RL task the motors of its many vehicles, through this one class.
"""

from __future__ import annotations

import warp as wp

__all__ = ["Motors"]


def _state_arrays(act_state) -> list[wp.array]:
    """The Warp arrays inside a composed ``newton.actuators`` Actuator.State, the delay + controller
    sub-states, in a construction-stable order, so two states built by the same ``actuator.state()``
    pair up positionally for the in-place copy-back.
    """
    out: list[wp.array] = []
    if act_state is None:
        return out
    for sub in (getattr(act_state, "delay_state", None), getattr(act_state, "controller_state", None)):
        if sub is None:
            continue
        for v in vars(sub).values():
            if isinstance(v, wp.array):
                out.append(v)
    return out


@wp.kernel
def _motor_mask(dofs: wp.array(dtype=wp.bool), indices: wp.array(dtype=wp.uint32), mask: wp.array(dtype=wp.bool)):
    """Each motor's entry of a mask over the model's degrees of freedom: the entry of the joint it drives."""
    i = wp.tid()
    mask[i] = dofs[int(indices[i])]


class Motors:
    """Every Newton motor a model holds, stepped together.

    Args:
        model: The finalized model. A model with no motors, such as a collapsed single body, steps nothing.
    """

    def __init__(self, model):
        self._actuators = list(getattr(model, "actuators", None) or [])
        self._states = [(a.state(), a.state()) for a in self._actuators]
        self._copies = [list(zip(_state_arrays(cur), _state_arrays(nxt), strict=True)) for cur, nxt in self._states]

    def __len__(self) -> int:
        return len(self._actuators)

    def step(self, state, control, dt: float) -> None:
        """Write each motor's torque, from ``state`` and the targets in ``control``, into ``control.joint_f``,
        zeroed first, and advance each motor's own state by ``dt``.
        """
        if not self._actuators:
            return
        control.joint_f.zero_()
        for actuator, (cur, nxt), copies in zip(self._actuators, self._states, self._copies, strict=True):
            actuator.step(state, control, cur, nxt, dt)
            for dst, src in copies:  # current ← next, in place: the graph-safe double-buffer
                wp.copy(dst, src)

    def reset(self, dofs: wp.array | None = None) -> None:
        """Zero the state of the motors that drive ``dofs``, a mask over the model's degrees of freedom, or of
        every motor when ``dofs`` is ``None``.
        """
        for actuator, (cur, _nxt) in zip(self._actuators, self._states, strict=True):
            if cur is None:
                continue
            mask = None
            if dofs is not None:
                mask = wp.empty(actuator.num_actuators, dtype=wp.bool, device=actuator.device)
                wp.launch(_motor_mask, dim=actuator.num_actuators, inputs=(dofs, actuator.indices), outputs=(mask,))
            cur.reset(mask)
