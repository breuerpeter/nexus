---
description: "The quadrotor vehicles nexus supports, resolved by name from the registry to a Universal Scene Description (USD) file: the Freefly Astro Max."
---

# Vehicles

The registry resolves vehicles by name, with `nexus run --vehicle <name>` and default
`astro-max`, to a USD. The framework only ever builds a model from a USD and never synthesizes
one in code. See [Conventions](../../design/conventions.md).

!!! warning
    nexus supports quadrotor vehicles only.

## Freefly Systems Astro Max

Astro is an industrial quadrotor for professional applications such as mapping and
inspection. Learn more on the [Freefly Systems](https://freeflysystems.com/) website.

The registry carries two Astro Max vehicles: `astro_max_base`, the base build, and
`astro_max_fpv`, which adds a First Person View (FPV) camera.

### Base

<!-- Interactive 3D preview, injected from registry.yaml by docs/hooks/vehicle_previews.py.
     Serves the .glb sibling of the vehicle's USD on CloudFront (or a local assets/local/
     copy when present, for preview-before-publish). -->

<!-- model-preview: astro_max_base -->

### FPV

<!-- model-preview: astro_max_fpv -->

!!! warning
    The simulation doesn't claim to accurately mimic the flight dynamics of Astro.
    If you are developing on Astro, don't fully rely on Software In The Loop (SITL)
    simulation to make conclusions about real-world behavior.
