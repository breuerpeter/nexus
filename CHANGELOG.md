# Changelog

## [0.2.0](https://github.com/breuerpeter/nexus/compare/nexus-sim-v0.1.0...nexus-sim-v0.2.0) (2026-10-09)


### ⚠ BREAKING CHANGES

* **recording:** Sim and Sim.from_orchestrator take no observe argument.
* **sensors:** Px4MavlinkController takes no gps_rate_hz; the GPS receiver's nexus:rate sets how often HIL_GPS goes out.
* nexus_sim.Measurement is gone, and a controller's exchange takes (t, timeout).
* a component schema that includes no role schema, or two, fails the build. A project's own schema includes one, such as NexusSensorRoleAPI.
* a vehicle declares NexusPx4API and NexusPx4SitlAPI on a Scope under its root prim, named Controller by convention; a vehicle that declares them on its root prim fails the build.
* Measurement has no state field; an estimator's estimate signal carries the vehicle's state.
* each concept has one word, defined once on a Concepts page ([#251](https://github.com/breuerpeter/nexus/issues/251))
* Sim.controller is gone; a flight reads a component it built from the object it holds.
* the hand-run ground stack and --stream go ([#242](https://github.com/breuerpeter/nexus/issues/242))
* **packaging:** import nexus_sim, not nexus.
* **guidance:** Sim and Sim.from_orchestrator take no reached_m, final_hold_s or operator; a flight passes guidance= and reads sim.guidance.
* **px4:** a script opens OffboardClient on the run's port map ([#204](https://github.com/breuerpeter/nexus/issues/204))
* **actuators:** the actuator seam splits into a command seam and a force seam ([#205](https://github.com/breuerpeter/nexus/issues/205))
* **config:** LaunchConfig loses six fields, and runtime.device rejects a device ordinal such as cuda:0.
* **actuators:** a vehicle USD declares each rotor's propeller with NexusPropellerAPI on the rotor's rigid body. A vehicle that authors propeller:* attributes on its rotor joints no longer builds.
* **build:** --px4, --px4-instance, Sim(px4=, px4_instance=), LaunchConfig.peers, Peers and Px4Peer are gone. A vehicle declares its PX4 SITL peer with NexusPx4SitlAPI, and a layer drops it to fly an external autopilot.
* **config:** a catalog with a defaults block no longer loads, and Sim and nexus run require a vehicle and a scene.
* **core:** Sim(control=), --control, LaunchConfig.control and set_control, and the catalog's px4 field are gone. A vehicle declares its controller in its USD.
* **core:** nexus.EnvSample is no longer exported.
* **cli:** --runtime leaves the command-line tool, LaunchConfig's runtime.backend and the empty isaacsim extra go, the isaacsim compose service and its image go, and nexus script runs only Kit-only asset scripts in the Kit image, with no nexus code inside.
* **config:** ship two Astro Max vehicles in the catalog ([#4](https://github.com/breuerpeter/nexus/issues/4))

### Features

* a component with no geometry sits on a scope prim of its own ([#259](https://github.com/breuerpeter/nexus/issues/259)) ([1cbc90f](https://github.com/breuerpeter/nexus/commit/1cbc90f0b84c2485f2fc0cab01d9ffa53897110a)), closes [#258](https://github.com/breuerpeter/nexus/issues/258)
* **actuators:** each rotor declares its propeller on its rigid body ([#166](https://github.com/breuerpeter/nexus/issues/166)) ([2668f83](https://github.com/breuerpeter/nexus/commit/2668f835f59b2e8493536537ec538568b4a9fca0)), closes [#32](https://github.com/breuerpeter/nexus/issues/32)
* **actuators:** the actuator seam splits into a command seam and a force seam ([#205](https://github.com/breuerpeter/nexus/issues/205)) ([1c9fb2c](https://github.com/breuerpeter/nexus/commit/1c9fb2c876b5bc24e43bff1065407ad5f35a6d57)), closes [#169](https://github.com/breuerpeter/nexus/issues/169)
* **assets:** a photoreal catalog scene with static geometry ([#153](https://github.com/breuerpeter/nexus/issues/153)) ([cda6cc7](https://github.com/breuerpeter/nexus/commit/cda6cc7533da1485c94217fc64634509015e3574)), closes [#101](https://github.com/breuerpeter/nexus/issues/101)
* **build:** a run composes an override layer, and the peer flags go ([#160](https://github.com/breuerpeter/nexus/issues/160)) ([b0e586a](https://github.com/breuerpeter/nexus/commit/b0e586adcaded90f66339295f5ab371d511e3a1c)), closes [#156](https://github.com/breuerpeter/nexus/issues/156)
* components declare typed signals, and the builder wires them once ([#249](https://github.com/breuerpeter/nexus/issues/249)) ([eafeb68](https://github.com/breuerpeter/nexus/commit/eafeb686a1cbb3872e76acd5044e85941b304112)), closes [#241](https://github.com/breuerpeter/nexus/issues/241)
* **config:** a project catalog extends the bundled one ([#17](https://github.com/breuerpeter/nexus/issues/17)) ([b520f74](https://github.com/breuerpeter/nexus/commit/b520f7406e2f715e7f8bfce6ce3959fc2aa0a2d4))
* **config:** a run names its vehicle and scene, and the catalog's defaults go ([#154](https://github.com/breuerpeter/nexus/issues/154)) ([e3fc6f0](https://github.com/breuerpeter/nexus/commit/e3fc6f024b7fb9963ee3ea0d7b90c9264c439c3e)), closes [#34](https://github.com/breuerpeter/nexus/issues/34)
* **config:** ship two Astro Max vehicles in the catalog ([#4](https://github.com/breuerpeter/nexus/issues/4)) ([32e1b2f](https://github.com/breuerpeter/nexus/commit/32e1b2f93aa9c2cbcf09ef7c2edd72f7399a0cab)), closes [#3](https://github.com/breuerpeter/nexus/issues/3)
* **core:** scene ambient values resolve at build, and the Environment seam goes ([#117](https://github.com/breuerpeter/nexus/issues/117)) ([087085c](https://github.com/breuerpeter/nexus/commit/087085cf941e89746cb58f6fc68b26e89046ae88)), closes [#38](https://github.com/breuerpeter/nexus/issues/38)
* **core:** the loop runs each component's work as device and host stages ([#120](https://github.com/breuerpeter/nexus/issues/120)) ([5180cd5](https://github.com/breuerpeter/nexus/commit/5180cd5835142408b83c01d05271f717c10f7e47)), closes [#39](https://github.com/breuerpeter/nexus/issues/39)
* **core:** the vehicle USD declares its controller, and --control goes ([#151](https://github.com/breuerpeter/nexus/issues/151)) ([4faa4d1](https://github.com/breuerpeter/nexus/commit/4faa4d1e9313a82291676148c709c84b2c3a21fa)), closes [#33](https://github.com/breuerpeter/nexus/issues/33)
* **docs:** the Benchmarking page plots each gated metric over the commits ([#201](https://github.com/breuerpeter/nexus/issues/201)) ([1ded2b9](https://github.com/breuerpeter/nexus/commit/1ded2b932218ee5ed59895c6b53a21fa8822e272)), closes [#81](https://github.com/breuerpeter/nexus/issues/81)
* each component's schema states its role ([#260](https://github.com/breuerpeter/nexus/issues/260)) ([cb4fa3e](https://github.com/breuerpeter/nexus/commit/cb4fa3e7acac472d2552bad5780083fdcf62a38d)), closes [#257](https://github.com/breuerpeter/nexus/issues/257)
* each sensor's output is a signal of its own type, cameras included ([#256](https://github.com/breuerpeter/nexus/issues/256)) ([10430fb](https://github.com/breuerpeter/nexus/commit/10430fb2807af51997c5efb3e22fb41ce9140ec8)), closes [#149](https://github.com/breuerpeter/nexus/issues/149)
* guidance and control read an estimator's estimate ([#252](https://github.com/breuerpeter/nexus/issues/252)) ([b4443f7](https://github.com/breuerpeter/nexus/commit/b4443f7879ddeb47a6e10a0273572ef9a28e6d7c)), closes [#228](https://github.com/breuerpeter/nexus/issues/228)
* **guidance:** guidance is a loop seam, the Operator protocol goes ([#203](https://github.com/breuerpeter/nexus/issues/203)) ([d825885](https://github.com/breuerpeter/nexus/commit/d8258856c013d4b786f6b614bf119e45762ef175)), closes [#210](https://github.com/breuerpeter/nexus/issues/210)
* **logging:** an entity path names its process, component and instance ([#193](https://github.com/breuerpeter/nexus/issues/193)) ([4d59000](https://github.com/breuerpeter/nexus/commit/4d59000182ef8a7dec9d6b98a0506b70af11290f)), closes [#47](https://github.com/breuerpeter/nexus/issues/47)
* **peers:** every peer ships a fake that speaks its link ([#155](https://github.com/breuerpeter/nexus/issues/155)) ([0125759](https://github.com/breuerpeter/nexus/commit/0125759a3449a7619d748b28b068fe7d75121bdc)), closes [#113](https://github.com/breuerpeter/nexus/issues/113)
* **px4:** a script opens OffboardClient on the run's port map ([#204](https://github.com/breuerpeter/nexus/issues/204)) ([f2ff238](https://github.com/breuerpeter/nexus/commit/f2ff238cbc0d891b7b3b1e3feb74cdb380ae1676)), closes [#194](https://github.com/breuerpeter/nexus/issues/194)
* **px4:** PX4 is a peer the run starts, built from its controller's pin ([#119](https://github.com/breuerpeter/nexus/issues/119)) ([618fb70](https://github.com/breuerpeter/nexus/commit/618fb70ad9c004dba785cc01caba08ed08e8ced5)), closes [#37](https://github.com/breuerpeter/nexus/issues/37)
* **recording:** the Recorder keeps each signal's history, and no component records itself ([#273](https://github.com/breuerpeter/nexus/issues/273)) ([fd098f7](https://github.com/breuerpeter/nexus/commit/fd098f7b47f616e8bc6ff8f596b940826be2e4ae)), closes [#250](https://github.com/breuerpeter/nexus/issues/250)
* **recording:** the recording holds every row, written from the Recorder's histories ([#264](https://github.com/breuerpeter/nexus/issues/264)) ([722ff3b](https://github.com/breuerpeter/nexus/commit/722ff3b07fb16030109ef2299a1f5136ac70d046)), closes [#111](https://github.com/breuerpeter/nexus/issues/111)
* **rendering:** the Kit peer runs NVIDIA's image as pulled ([#126](https://github.com/breuerpeter/nexus/issues/126)) ([32ebe1c](https://github.com/breuerpeter/nexus/commit/32ebe1cb462254877c96b0327f91a10bc9de5fb2)), closes [#124](https://github.com/breuerpeter/nexus/issues/124)
* **sensors:** sensors are declared by API schemas in the vehicle USD ([#180](https://github.com/breuerpeter/nexus/issues/180)) ([1eed48b](https://github.com/breuerpeter/nexus/commit/1eed48b852b83fcface35eea18e6046e8a159581)), closes [#9](https://github.com/breuerpeter/nexus/issues/9)
* **sensors:** the IMU honours its prim's full transform on any body ([#212](https://github.com/breuerpeter/nexus/issues/212)) ([c7bf2bd](https://github.com/breuerpeter/nexus/commit/c7bf2bda8698b9e3b0101ff3192de7b3ecffe821)), closes [#52](https://github.com/breuerpeter/nexus/issues/52)
* **sensors:** the in-graph sensors sample at their declared rates ([#223](https://github.com/breuerpeter/nexus/issues/223)) ([9d74670](https://github.com/breuerpeter/nexus/commit/9d746704316fd3e6cdc12de14646eb6ae28b098a)), closes [#51](https://github.com/breuerpeter/nexus/issues/51)
* the hand-run ground stack and --stream go ([#242](https://github.com/breuerpeter/nexus/issues/242)) ([26c6b61](https://github.com/breuerpeter/nexus/commit/26c6b617f112b95c6e579b72389e5366e85990a6)), closes [#42](https://github.com/breuerpeter/nexus/issues/42)
* **usd:** a schema plugin and an open registry for components ([#140](https://github.com/breuerpeter/nexus/issues/140)) ([8d6d494](https://github.com/breuerpeter/nexus/commit/8d6d4940aac32f5b70b2ae548c87b7487599fb92)), closes [#31](https://github.com/breuerpeter/nexus/issues/31)
* **usd:** schemas evolve by version, and the build fails an unknown one ([#176](https://github.com/breuerpeter/nexus/issues/176)) ([c79c3f2](https://github.com/breuerpeter/nexus/commit/c79c3f2e7e0b947b6d37fff02b7745a45ef6fb59)), closes [#114](https://github.com/breuerpeter/nexus/issues/114)


### Bug Fixes

* **assets:** a scene re-root keeps material bindings and shader links ([#128](https://github.com/breuerpeter/nexus/issues/128)) ([1f238f1](https://github.com/breuerpeter/nexus/commit/1f238f160c13779e77505a415a86e819a6310358)), closes [#65](https://github.com/breuerpeter/nexus/issues/65)
* **ci:** a runner cancelled mid-start no longer outlives its run ([#136](https://github.com/breuerpeter/nexus/issues/136)) ([a2b0d9b](https://github.com/breuerpeter/nexus/commit/a2b0d9b5c88bb73bae7be776cf54935b7bfa2a34)), closes [#78](https://github.com/breuerpeter/nexus/issues/78)
* **ci:** an upload run holds back a red flight's docs recording ([#197](https://github.com/breuerpeter/nexus/issues/197)) ([2e439e4](https://github.com/breuerpeter/nexus/commit/2e439e4af60532ad663231c8bb0ce611d9efe93f)), closes [#95](https://github.com/breuerpeter/nexus/issues/95)
* **ci:** gate the rl leg by the kind of quantity each check reads ([#68](https://github.com/breuerpeter/nexus/issues/68)) ([ab8dc8a](https://github.com/breuerpeter/nexus/commit/ab8dc8a42544f29c63ab5f9e5615cdbc73510449)), closes [#15](https://github.com/breuerpeter/nexus/issues/15)
* **ci:** let an approved fork run check out the fork's code ([#24](https://github.com/breuerpeter/nexus/issues/24)) ([7ab3a28](https://github.com/breuerpeter/nexus/commit/7ab3a2810ea317d87ddf729cac5173ae456a53f3))
* **ci:** lint reds a pull request that stales nexus-rl/uv.lock ([#207](https://github.com/breuerpeter/nexus/issues/207)) ([a984b80](https://github.com/breuerpeter/nexus/commit/a984b80b32eaae098c55fecd24107ad70ac43d08)), closes [#202](https://github.com/breuerpeter/nexus/issues/202)
* **ci:** the bench feed reads under a ListBucket grant, and the runner says so ([#138](https://github.com/breuerpeter/nexus/issues/138)) ([80bc50c](https://github.com/breuerpeter/nexus/commit/80bc50cdd1e9e724e4d1c7a962765646dd429005)), closes [#116](https://github.com/breuerpeter/nexus/issues/116)
* **ci:** the release pull request keeps both locks fresh ([#183](https://github.com/breuerpeter/nexus/issues/183)) ([f63336b](https://github.com/breuerpeter/nexus/commit/f63336b788d0f380879a5fa70d18887f424a1aa4))
* **cli:** launch the Kit container from an installed package ([#22](https://github.com/breuerpeter/nexus/issues/22)) ([821a007](https://github.com/breuerpeter/nexus/commit/821a0073bd096671066e3311de3054b19cf09cbe)), closes [#12](https://github.com/breuerpeter/nexus/issues/12)
* **config:** the receipt records only what a run applies ([#184](https://github.com/breuerpeter/nexus/issues/184)) ([02ea594](https://github.com/breuerpeter/nexus/commit/02ea594178134f14285c61f2502af9ee232e0e57)), closes [#173](https://github.com/breuerpeter/nexus/issues/173)
* **docs:** the docs and the config name only what the tree holds ([#225](https://github.com/breuerpeter/nexus/issues/225)) ([2f6c819](https://github.com/breuerpeter/nexus/commit/2f6c8193021e821dafd7c0b23aff468c2d95b004)), closes [#220](https://github.com/breuerpeter/nexus/issues/220)
* **examples:** goto_policy fetches its hosted policy from the catalog base ([#67](https://github.com/breuerpeter/nexus/issues/67)) ([1445f6a](https://github.com/breuerpeter/nexus/commit/1445f6a460123da7bde2ef1f75f31324e2818819)), closes [#66](https://github.com/breuerpeter/nexus/issues/66)
* **examples:** the acados extra names only packages PyPI serves ([#192](https://github.com/breuerpeter/nexus/issues/192)) ([872045d](https://github.com/breuerpeter/nexus/commit/872045d17067f27d2fc586fee85c4bd915848248)), closes [#182](https://github.com/breuerpeter/nexus/issues/182)
* **examples:** the PX4 gate forgives a logger cap line for any topic ([#141](https://github.com/breuerpeter/nexus/issues/141)) ([3068275](https://github.com/breuerpeter/nexus/commit/3068275ace16eade79331dd6ca9708b23c6b9bf0)), closes [#137](https://github.com/breuerpeter/nexus/issues/137)
* **notices:** name every file a foreign header marks; goto_env is Apache-2.0 ([#227](https://github.com/breuerpeter/nexus/issues/227)) ([dc292cb](https://github.com/breuerpeter/nexus/commit/dc292cbce1d52be11291cd05ff7322db64b2bdc4)), closes [#219](https://github.com/breuerpeter/nexus/issues/219)
* **orchestrator:** capture the host-exchange graph before PX4 dials in ([#61](https://github.com/breuerpeter/nexus/issues/61)) ([94009bc](https://github.com/breuerpeter/nexus/commit/94009bc1ba3c738f720bf4e7dbec7b840eda0d88)), closes [#26](https://github.com/breuerpeter/nexus/issues/26)
* **packaging:** an installed nexus-sim pins the warp-lang CI flies ([#127](https://github.com/breuerpeter/nexus/issues/127)) ([cec9165](https://github.com/breuerpeter/nexus/commit/cec9165f74278d0880bba13230862a73e84d9f8f)), closes [#62](https://github.com/breuerpeter/nexus/issues/62)
* **packaging:** the wheel and sdist ship no CLAUDE.md ([#240](https://github.com/breuerpeter/nexus/issues/240)) ([7613209](https://github.com/breuerpeter/nexus/commit/7613209f8c8abf5a6c7e09035de530d84894fe56)), closes [#217](https://github.com/breuerpeter/nexus/issues/217)
* **px4:** a late PX4 dial-in logs no ekf2 missing data line ([#60](https://github.com/breuerpeter/nexus/issues/60)) ([cab8ef9](https://github.com/breuerpeter/nexus/commit/cab8ef92bd31469d5666679aa6417a3322b3ee69)), closes [#57](https://github.com/breuerpeter/nexus/issues/57)
* **px4:** allowlist the missing-parameter message, not [param] ([#23](https://github.com/breuerpeter/nexus/issues/23)) ([eb78bb9](https://github.com/breuerpeter/nexus/commit/eb78bb9a9e9f8a973a8a28c62b4e1d9caa2e6e74)), closes [#21](https://github.com/breuerpeter/nexus/issues/21)


### Performance Improvements

* **tests:** fixture-vehicle tests share their builds, and the Kit fake stops at once ([#232](https://github.com/breuerpeter/nexus/issues/232)) ([9224844](https://github.com/breuerpeter/nexus/commit/9224844a53404da8245afbabec55f765edd3b92b)), closes [#209](https://github.com/breuerpeter/nexus/issues/209)


### Code Refactoring

* each concept has one word, defined once on a Concepts page ([#251](https://github.com/breuerpeter/nexus/issues/251)) ([6d50ebb](https://github.com/breuerpeter/nexus/commit/6d50ebb16841db08a151acb90d8dd2ecf3fb7c49)), closes [#247](https://github.com/breuerpeter/nexus/issues/247)
* **packaging:** the import package is nexus_sim, imported as nx ([#230](https://github.com/breuerpeter/nexus/issues/230)) ([c77617c](https://github.com/breuerpeter/nexus/commit/c77617c11c8e85f4231af561e8cb0e2d46fb2a94)), closes [#216](https://github.com/breuerpeter/nexus/issues/216)

## Changelog

All notable changes to the project are documented here. This file is maintained automatically by [release-please](https://github.com/googleapis/release-please) from [Conventional Commit](https://www.conventionalcommits.org/) messages on `main`; release entries are prepended below as versions are tagged.
