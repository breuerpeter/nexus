---
description: "API reference for the component contract, the stage and signal types it speaks, the registry that maps a declaration to its class, and the peer contract."
---

# Components, registry, and peers

```python
from nexus_uav import Component, Stage, Tick, Signal, Registry, register, Peer
```

One contract serves every [role](../../design/concepts.md#role): a component subclasses `Component`,
states its per-tick work as a list of `Stage`, each declaring the `Signal`s it reads and writes, and
overrides the lifecycle members it has. The loop runs each stage over the shared `Tick`. A
[registry](../../design/concepts.md#registry) maps each schema a vehicle applies to the class that
builds it, and a [peer](../../design/concepts.md#peer) is a process the run starts and stops on the
`Peer` contract. [Bringing your own component](../../guide/your-own-component.md) walks through all four.

::: nexus_uav.Component
::: nexus_uav.Stage
::: nexus_uav.Tick
::: nexus_uav.Signal
::: nexus_uav.Registry
::: nexus_uav.register
::: nexus_uav.Peer
