---
description: "The component schemas the nexus Universal Scene Description (USD) plugin defines: what each applies to, and each attribute's type, default, units and range."
---

# Schemas

A component schema is an applied API schema whose attributes sit in the `nexus:` namespace. The build resolves each one a prim applies to its class, and passes each attribute as a keyword argument of the same name in snake case: `nexus:accNoise` becomes `acc_noise`. An attribute the prim doesn't author takes the default below.

This page comes from the schema plugin that ships with nexus, so it lists what the build accepts. An authored `nexus:` attribute that no applied schema defines fails the build and names the prim.

A schema changes only by a new version, which OpenUSD names with a suffix: `NexusImuAPI`, then `NexusImuAPI_1`. A version can gain an attribute with a default. A renamed or removed attribute, or a new type or unit, makes a new version, and its section below states what changed. A prim that applies a version this release doesn't define fails the build, and the message names the prim, the version, and the nexus version.

<!-- schema-reference -->
