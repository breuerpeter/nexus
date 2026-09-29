---
description: "The component schemas the nexus Universal Scene Description (USD) plugin defines: what each applies to, and each attribute's type, default, units and range."
---

# Schemas

A component schema is an applied API schema whose attributes sit in the `nexus:` namespace. The build resolves each one a prim applies to its class, and passes each attribute as a keyword argument of the same name in snake case: `nexus:accNoise` becomes `acc_noise`. An attribute the prim doesn't author takes the default below.

This page comes from the schema plugin that ships with nexus, so it lists what the build accepts. An authored `nexus:` attribute that no applied schema defines fails the build and names the prim.

<!-- schema-reference -->
