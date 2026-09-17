---
description: "The real drone system organized into six subsystem planes, and the substitution boundary where real drone code stops and nexus's simulated components take over."
---

# Real system ↔ simulation

nexus's primary job is to reproduce as much of the **real drone system** as possible in CI,
substituting only what it has to simulate. The real system and the sim are both organized into the
**same six subsystem planes**, so the two line up box-for-box and their difference *is* the
substitution boundary.

## The substitution boundary

The guiding principle is to **run as much real drone-system code as possible**: real PX4 firmware,
the real Ground Control Station (GCS) app, real onboard apps. The sim stands in only for the
physical layer and the hardware transports. Draw the boundary as a horizontal line: real code on
top, simulated components below, interfaces crossing it. Anything real that lacks a defined sim
counterpart is a tracked gap.

| Plane | Role | Sim treatment |
|---|---|---|
| **Physical** | airframe, sensors, actuators, power, payload hardware, the world | **Simulated**: the *only* plane the sim replaces |
| **Autopilot** | PX4 on the Flight Management Unit (FMU) | **Real, as-is** via PX4 Software In The Loop (SITL): the boundary cuts *through* the FMU |
| **Companion** | the system-on-module and its services | **Real, emulated**: the companion image under an emulator |
| **Links** | radios, Wi-Fi, LTE | **Network-emulated**: bridged or loopback with added latency and loss |
| **Operator** | controller, GCS, mobile-device apps | **Real apps**: scripted, real hardware, or the browser build depending on the app |
| **Cloud** | NTRIP, Suite, fleet | **Real services or local doubles** |

## Where the boundary cuts

Two subtleties don't fit a clean plane split, because they cut *through* a device:

- **Inside the FMU.** PX4's *core* is real-as-is in SITL and is the system under test: the
  estimator, controllers, navigator, commander, MAVLink, and logger. Its NuttX *drivers* and the IO
  board **don't exist in SITL**, so the sim replaces them at the `HIL_*` ingestion layer. Those
  drivers cover the sensors, Global Positioning System (GPS), Electronic Speed Controller (ESC),
  smart battery, and rangefinder. When a test asserts on a driver's output, such as an ESC or
  smart-battery status topic, the simulated component must emit that topic, not just the underlying
  physical quantity.
- **RC has two real paths.** SBUS through the IO board, and `MANUAL_CONTROL` over MAVLink. SITL
  removes the IO-board path, so the sim injects RC either as MAVLink manual control, the real path
  for the Pilot Pro app and GCS, or as simulated RC. The choice is per scenario, since
  operator-training fidelity depends on it.

## What each app needs

The planes each app exercises differ, which is what makes them incremental milestones rather
than one monolith:

| App | Physical | Autopilot | Companion | Operator | Cloud |
|---|---|---|---|---|---|
| **SITL testing in CI** ★ | simulated | PX4 SITL | emulated companion | scripted GCS or harness | test management + log pull |
| **Pilot training** | simulated | PX4 SITL | companion | real Pilot Pro hardware | cloud VM + streaming |
| **Customer preview** | simulated | PX4 SITL | companion | browser GCS + joystick | Suite import + NTRIP |
| **Flight-stack development** | simulated | PX4 SITL | optional | API client | none |
| **Design optimization** | simulated | in-process differentiable Proportional Integral Derivative (PID) | none | none | none |

The sim replaces the physical plane in every case. The rest of the stack runs progressively more real
code as an app moves from batch CI toward a human flying the sim from real hardware. The items
still to build to complete each app are the companion emulation, link emulation, streaming,
and Suite round-trip.
