---
description: "API reference for the estimator, the role that writes the estimate the guidance and the controllers read: GroundTruthEstimator, the passthrough, and the PoseTwist type it writes."
---

# Estimator

The **estimator** is the first block of the control cascade: each tick it turns the sensors' values
into the estimate, the vehicle's state that the guidance and the controller read. It's a component of
the loop, and its stages run after the sensors' and before the guidance's. The estimate is the
`estimate` [signal](../../design/execution.md#signals), of the type `PoseTwist`: the base body's
position, its quaternion, and its linear and angular velocity, in world axes. The guidance and each
controller that reads the vehicle's state read it, never the physics state, so a controller written
against the estimate flies behind any estimator.

One estimator ships, `GroundTruthEstimator`, a passthrough that hands on the base body's true pose and
twist with no noise and no delay. Its stage also runs once over the settled state before the first
tick, so the guidance's first stage reads an estimate. A run whose guidance or controller reads the
estimate, and that has no estimator, fails before any stage runs, with an error that names the readers
and the estimator. PX4 takes none, because its own estimator runs in its peer and takes the sensors'
values over the lockstep link.

The estimator belongs to the vehicle's flight stack, beside its controller. Each example's assembly
constructs it and hands it to the loop. The classes are **internal**: an example imports them from
`nexus_sim._src`, not from the top-level package.

```python
orch = Orchestrator(..., estimator=GroundTruthEstimator(), controller=controller)
```

::: nexus_sim._src.vehicle.estimators.ground_truth.GroundTruthEstimator
::: nexus_sim._src.core.schema.PoseTwist
