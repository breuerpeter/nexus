---
description: "The component schemas the nexus Universal Scene Description (USD) plugin defines: what each applies to, and each attribute's type, default, units and range."
---

# Schemas

A component schema is an applied API schema whose attributes sit in the `nexus:` namespace. The build resolves each one a prim applies to its class, and passes each attribute as a keyword argument of the same name in snake case: `nexus:accNoise` becomes `acc_noise`. An attribute the prim doesn't author takes the default below.

Each component schema states the role its component fills: it includes a role schema, such as `NexusSensorRoleAPI`, as a built-in, and its section below names that role. The role places the component. A sensor sits on its mount prim under the rigid body it rides. The controller and the estimator each sit on a `Scope` of its own whose parent is the vehicle's root prim. A force element, such as a propeller, sits where its own schema says. A schema that states no role, or two, fails the build and names the prim and the schema. A project's own component schema includes a role schema the same way. A peer's schema, such as `NexusPx4SitlAPI`, states no role, since it defines no attribute.

This page comes from the schema plugin that ships with nexus, so it lists what the build accepts. An authored `nexus:` attribute that no applied schema defines fails the build and names the prim.

A schema changes only by a new version, which OpenUSD names with a suffix: `NexusImuAPI`, then `NexusImuAPI_1`. A version can gain an attribute with a default. A renamed or removed attribute, or a new type, default or unit, makes a new version, and its section below states what changed. A prim that applies a version this release doesn't define fails the build, and the message names the prim, the version, and the nexus version.

<!-- schema-reference -->
