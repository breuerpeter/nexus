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

Astro Max ships in a base build and with payloads, each built around a
First Person View (FPV) camera. Each is a registry vehicle of its own, named `astro_max_base`,
`astro_max_fpv`, `astro_max_fpv_lr1`, `astro_max_fpv_flux`, or `astro_max_fpv_wiris`:

=== "Base"

    <!-- Interactive 3D preview, injected from registry.yaml by docs/hooks/vehicle_previews.py.
         Serves the .glb sibling of the vehicle's USD on CloudFront (or a local assets/local/
         copy when present, for preview-before-publish). -->

    <!-- model-preview: astro_max_base -->

=== "With FPV camera"

    <!-- model-preview: astro_max_fpv -->

=== "FPV and LR1 payload"

    <!-- model-preview: astro_max_fpv_lr1 -->

=== "FPV and Flux LiDAR payload"

    <!-- model-preview: astro_max_fpv_flux -->

=== "FPV and Wiris IR payload"

    <!-- model-preview: astro_max_fpv_wiris -->

    The IR camera is a thermal sensor, not a second colour one: it renders the scene's
    emission as an LWIR image and is published on its own `ir1` stream, advertised THERMAL
    so the Ground Control Station (GCS) shows it beside the EO feed. The name reflects the
    camera the payload represents; it is not a product-accurate model of the Workswell Wiris.

!!! warning
    The simulation doesn't claim to accurately mimic the flight dynamics of Astro.
    If you are developing on Astro, don't fully rely on Software In The Loop (SITL)
    simulation to make conclusions about real-world behavior.
