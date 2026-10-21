# Changelog

## [0.6.1](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.6.0...nexus-uav-v0.6.1) (2026-10-21)


### Bug Fixes

* **peers:** a run keeps PX4's store across a restart unless asked to start clean ([#355](https://github.com/breuerpeter/nexus/issues/355)) ([9463732](https://github.com/breuerpeter/nexus/commit/9463732db422b6c26f07e5d90f463ba723bd5acc)), closes [#354](https://github.com/breuerpeter/nexus/issues/354)

## [0.6.0](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.5.0...nexus-uav-v0.6.0) (2026-10-20)


### Features

* **ci:** a labelled PR touching a peer's folder flies that peer real ([#337](https://github.com/breuerpeter/nexus/issues/337)) ([b9d4eea](https://github.com/breuerpeter/nexus/commit/b9d4eeabe952915a184ff9673c3c618af3490ba2)), closes [#162](https://github.com/breuerpeter/nexus/issues/162)
* **config:** a catalog vehicle entry composes a base vehicle and its layers ([#346](https://github.com/breuerpeter/nexus/issues/346)) ([877eae2](https://github.com/breuerpeter/nexus/commit/877eae20c1d7bf60e3055492d31f36e17f1000a9)), closes [#331](https://github.com/breuerpeter/nexus/issues/331)
* **peers:** a declared peer's link reads the signals it names ([#347](https://github.com/breuerpeter/nexus/issues/347)) ([03f82ae](https://github.com/breuerpeter/nexus/commit/03f82ae7e1166db455533214893e5c7b556e2e7f)), closes [#341](https://github.com/breuerpeter/nexus/issues/341)
* **peers:** the run joins a declared peer and PX4 SITL over a serial link it owns ([#345](https://github.com/breuerpeter/nexus/issues/345)) ([60949e1](https://github.com/breuerpeter/nexus/commit/60949e11b12445280360f2bae5156d7a5dcdfd85)), closes [#342](https://github.com/breuerpeter/nexus/issues/342)
* **vehicle:** each rotor flies its own propeller values, not one shared set ([#348](https://github.com/breuerpeter/nexus/issues/348)) ([f92a746](https://github.com/breuerpeter/nexus/commit/f92a746567262db3fadd17c9d929e04198868e82)), closes [#185](https://github.com/breuerpeter/nexus/issues/185)


### Bug Fixes

* **peers:** a package's pin file for a peer it registers breaks no run ([#352](https://github.com/breuerpeter/nexus/issues/352)) ([a35b24a](https://github.com/breuerpeter/nexus/commit/a35b24a5a193d2058b86b06a7e79f0c430cfeaa5)), closes [#316](https://github.com/breuerpeter/nexus/issues/316)

## [0.5.0](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.4.0...nexus-uav-v0.5.0) (2026-10-19)


### ⚠ BREAKING CHANGES

* **peers:** a run's folder holds `px4_sitl/console.log` and `px4_sitl/<date>/<time>.ulg` in place of `px4.log` and `ulog/`, and `kit/console.log` in place of `kit.log`.

### Features

* **peers:** a peer states how it keeps time, and a run that can't keep it fails first ([#336](https://github.com/breuerpeter/nexus/issues/336)) ([317ebfb](https://github.com/breuerpeter/nexus/commit/317ebfb8da8af7e42d79853618cdcf1ebbd8e654)), closes [#325](https://github.com/breuerpeter/nexus/issues/325)
* **peers:** a project's container peer subclasses the public container peer ([#333](https://github.com/breuerpeter/nexus/issues/333)) ([9cab39e](https://github.com/breuerpeter/nexus/commit/9cab39e8a6b1d09b3551a5015d9bdb50d156cadd)), closes [#306](https://github.com/breuerpeter/nexus/issues/306)


### Bug Fixes

* **px4:** a run boots PX4 with no state an earlier run left ([#328](https://github.com/breuerpeter/nexus/issues/328)) ([b5f43e5](https://github.com/breuerpeter/nexus/commit/b5f43e5ba2d24823a7aea4d676eed99c893b8562)), closes [#327](https://github.com/breuerpeter/nexus/issues/327)

## [0.4.0](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.3.0...nexus-uav-v0.4.0) (2026-10-15)


### ⚠ BREAKING CHANGES

* NewtonPhysics takes no step_actuators, and the examples library drops Rotors and rigid_body_wrench_batched. The nexus-rl GoTo env still imports the latter until it moves to the articulated plant.
* **core:** a run with no verdict raises the public RunError ([#324](https://github.com/breuerpeter/nexus/issues/324))
* **build:** NexusPx4API is retired for NexusPx4API_1, which authors no airframe; nexus:airframe is an attribute of NexusPx4SitlAPI, declared on a Scope of its own, and a layer drops the peer by deactivating that Scope. build_from_launch takes no peers mapping, the receipt's px4 field is gone, Px4MavlinkController takes the run's instance in place of port and target_system, and KitPeer takes its pin in place of an image.

### Features

* **api:** a run writes its files to the folder the caller names ([#323](https://github.com/breuerpeter/nexus/issues/323)) ([2a6617b](https://github.com/breuerpeter/nexus/commit/2a6617bec24d528640ed25d224bbe410dc554691)), closes [#320](https://github.com/breuerpeter/nexus/issues/320)
* **assets:** s3:// assets come through boto3 under nexus-uav[s3] ([#326](https://github.com/breuerpeter/nexus/issues/326)) ([a4d9d62](https://github.com/breuerpeter/nexus/commit/a4d9d625030de5945d32e4554b5599620b946d5c)), closes [#322](https://github.com/breuerpeter/nexus/issues/322)
* **build:** every role and peer builds from the vehicle's USD ([#315](https://github.com/breuerpeter/nexus/issues/315)) ([46764ac](https://github.com/breuerpeter/nexus/commit/46764acaaf9f8c49c1673b640cc496f77833e643)), closes [#261](https://github.com/breuerpeter/nexus/issues/261)
* **core:** a run with no verdict raises the public RunError ([#324](https://github.com/breuerpeter/nexus/issues/324)) ([2c2c1d9](https://github.com/breuerpeter/nexus/commit/2c2c1d9913cf85571a10aa104f15d246523d7067)), closes [#321](https://github.com/breuerpeter/nexus/issues/321)
* every controller flies one plant on the vehicle's Newton motors ([#319](https://github.com/breuerpeter/nexus/issues/319)) ([1c6afe6](https://github.com/breuerpeter/nexus/commit/1c6afe6c145987072e09fcf9db3af4ed689957c8))

## [0.3.0](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.2.0...nexus-uav-v0.3.0) (2026-10-13)


### ⚠ BREAKING CHANGES

* **catalog:** an installed package adds a catalog layer and its pin files ([#313](https://github.com/breuerpeter/nexus/issues/313))

### Features

* **catalog:** an installed package adds a catalog layer and its pin files ([#313](https://github.com/breuerpeter/nexus/issues/313)) ([cf422f4](https://github.com/breuerpeter/nexus/commit/cf422f469b7f6e24466306f6af16176d6a64b280)), closes [#305](https://github.com/breuerpeter/nexus/issues/305)

## [0.2.0](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.1.0...nexus-uav-v0.2.0) (2026-10-13)


### ⚠ BREAKING CHANGES

* **peers:** PX4_DIR goes; set NEXUS_PX4_SITL_REF to a checkout's absolute path instead. A project's nexus.px4.ref becomes nexus.px4_sitl.ref, and the old name now fails the run.

### Features

* **peers:** every peer resolves its pin the same way ([#309](https://github.com/breuerpeter/nexus/issues/309)) ([3b2b879](https://github.com/breuerpeter/nexus/commit/3b2b8794751b834f6272807f6b68da5a011bc586)), closes [#304](https://github.com/breuerpeter/nexus/issues/304)


### Bug Fixes

* **loop:** a run whose PX4 never starts lockstep fails ([#308](https://github.com/breuerpeter/nexus/issues/308)) ([b6e5b57](https://github.com/breuerpeter/nexus/commit/b6e5b57ed536adb588ac6846c58b70acfef389ff)), closes [#129](https://github.com/breuerpeter/nexus/issues/129)
* **px4:** a run's artifacts name the ULog PX4 wrote for that run ([#310](https://github.com/breuerpeter/nexus/issues/310)) ([dc13009](https://github.com/breuerpeter/nexus/commit/dc13009b5e30c5ec686ffcf4d68bf063ddefb7dd)), closes [#145](https://github.com/breuerpeter/nexus/issues/145)

## [0.1.0](https://github.com/breuerpeter/nexus/compare/nexus-uav-v0.1.0...nexus-uav-v0.1.0) (2026-10-12)


### ⚠ BREAKING CHANGES

* the import package is nexus_uav and the distribution nexus-uav.
* **sensors:** every sensor decimates by one due-time rule ([#294](https://github.com/breuerpeter/nexus/issues/294))
* **api:** History.latest() is gone, and Signal.read gives a struct instance or an array without the leading axis.
* **sensors:** Component.connect takes the links of the peers the class requires; SensorRun, Site and the scene's origin reader are gone, and a sensor's class takes the model and its site.
* **scene:** --geo, Sim(geo=), LaunchConfig.set_geodetic_origin and the catalog's and receipt's geodetic_origin are gone, and a catalog scene names its USD file. An override layer moves a scene's origin.
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
* **api:** a script reads a signal's live value by its path, and the sim time ([#287](https://github.com/breuerpeter/nexus/issues/287)) ([8d3e5e4](https://github.com/breuerpeter/nexus/commit/8d3e5e4e54752cfad106c9d939bd93d7fe99b3af)), closes [#280](https://github.com/breuerpeter/nexus/issues/280)
* **api:** export the component, registry and peer contracts ([#277](https://github.com/breuerpeter/nexus/issues/277)) ([2ce4b9b](https://github.com/breuerpeter/nexus/commit/2ce4b9bdc96237df47b9e6f6150a1c05f0e4e678)), closes [#44](https://github.com/breuerpeter/nexus/issues/44)
* **api:** the RL project imports only the public surface ([#279](https://github.com/breuerpeter/nexus/issues/279)) ([80d4036](https://github.com/breuerpeter/nexus/commit/80d4036da40a84de8b196cf2f5b257383138c491)), closes [#53](https://github.com/breuerpeter/nexus/issues/53)
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
* **loop:** each tick has one exact sim time, and its rows describe it ([#290](https://github.com/breuerpeter/nexus/issues/290)) ([5122ce1](https://github.com/breuerpeter/nexus/commit/5122ce12b00f33a0eaca353e8045839377120184)), closes [#229](https://github.com/breuerpeter/nexus/issues/229)
* **peers:** every peer ships a fake that speaks its link ([#155](https://github.com/breuerpeter/nexus/issues/155)) ([0125759](https://github.com/breuerpeter/nexus/commit/0125759a3449a7619d748b28b068fe7d75121bdc)), closes [#113](https://github.com/breuerpeter/nexus/issues/113)
* **px4:** a script opens OffboardClient on the run's port map ([#204](https://github.com/breuerpeter/nexus/issues/204)) ([f2ff238](https://github.com/breuerpeter/nexus/commit/f2ff238cbc0d891b7b3b1e3feb74cdb380ae1676)), closes [#194](https://github.com/breuerpeter/nexus/issues/194)
* **px4:** PX4 is a peer the run starts, built from its controller's pin ([#119](https://github.com/breuerpeter/nexus/issues/119)) ([618fb70](https://github.com/breuerpeter/nexus/commit/618fb70ad9c004dba785cc01caba08ed08e8ced5)), closes [#37](https://github.com/breuerpeter/nexus/issues/37)
* **recording:** the plant writes signals, and every history reads as records ([#282](https://github.com/breuerpeter/nexus/issues/282)) ([149be06](https://github.com/breuerpeter/nexus/commit/149be06213bce601d3ec2137da3e5b12ddd6d048)), closes [#275](https://github.com/breuerpeter/nexus/issues/275)
* **recording:** the Recorder keeps each signal's history, and no component records itself ([#273](https://github.com/breuerpeter/nexus/issues/273)) ([fd098f7](https://github.com/breuerpeter/nexus/commit/fd098f7b47f616e8bc6ff8f596b940826be2e4ae)), closes [#250](https://github.com/breuerpeter/nexus/issues/250)
* **recording:** the recording holds every row, written from the Recorder's histories ([#264](https://github.com/breuerpeter/nexus/issues/264)) ([722ff3b](https://github.com/breuerpeter/nexus/commit/722ff3b07fb16030109ef2299a1f5136ac70d046)), closes [#111](https://github.com/breuerpeter/nexus/issues/111)
* **rendering:** the Kit peer runs NVIDIA's image as pulled ([#126](https://github.com/breuerpeter/nexus/issues/126)) ([32ebe1c](https://github.com/breuerpeter/nexus/commit/32ebe1cb462254877c96b0327f91a10bc9de5fb2)), closes [#124](https://github.com/breuerpeter/nexus/issues/124)
* **scene:** a scene's PhysicsScene prim holds its geodetic origin ([#284](https://github.com/breuerpeter/nexus/issues/284)) ([9dca7ac](https://github.com/breuerpeter/nexus/commit/9dca7ac962db63b2c6885a33a7dc89713278deb6)), closes [#281](https://github.com/breuerpeter/nexus/issues/281)
* **sensors:** a sensor's prim is a Newton site, and SensorRun goes ([#283](https://github.com/breuerpeter/nexus/issues/283)) ([35b16ff](https://github.com/breuerpeter/nexus/commit/35b16ff079288e2171519c3f095a8ab9c6b8a8ba)), closes [#278](https://github.com/breuerpeter/nexus/issues/278)
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
* **sensors:** every sensor decimates by one due-time rule ([#294](https://github.com/breuerpeter/nexus/issues/294)) ([f964a1b](https://github.com/breuerpeter/nexus/commit/f964a1bca2f7934eb44cb8e6292c0437298acf0d)), closes [#263](https://github.com/breuerpeter/nexus/issues/263)


### Performance Improvements

* **tests:** fixture-vehicle tests share their builds, and the Kit fake stops at once ([#232](https://github.com/breuerpeter/nexus/issues/232)) ([9224844](https://github.com/breuerpeter/nexus/commit/9224844a53404da8245afbabec55f765edd3b92b)), closes [#209](https://github.com/breuerpeter/nexus/issues/209)


### Miscellaneous Chores

* **release:** the first release is 0.1.0 and its README links resolve ([#295](https://github.com/breuerpeter/nexus/issues/295)) ([2049b5e](https://github.com/breuerpeter/nexus/commit/2049b5ecc4c3fa023429b41f1e22195bdfab4478)), closes [#55](https://github.com/breuerpeter/nexus/issues/55)


### Code Refactoring

* each concept has one word, defined once on a Concepts page ([#251](https://github.com/breuerpeter/nexus/issues/251)) ([6d50ebb](https://github.com/breuerpeter/nexus/commit/6d50ebb16841db08a151acb90d8dd2ecf3fb7c49)), closes [#247](https://github.com/breuerpeter/nexus/issues/247)
* **packaging:** the import package is nexus_sim, imported as nx ([#230](https://github.com/breuerpeter/nexus/issues/230)) ([c77617c](https://github.com/breuerpeter/nexus/commit/c77617c11c8e85f4231af561e8cb0e2d46fb2a94)), closes [#216](https://github.com/breuerpeter/nexus/issues/216)


### Build System

* publish as nexus-uav with the import package nexus_uav ([#297](https://github.com/breuerpeter/nexus/issues/297)) ([0972f33](https://github.com/breuerpeter/nexus/commit/0972f332f0bb335ce0e3ed5e574874a889b8ad89)), closes [#296](https://github.com/breuerpeter/nexus/issues/296)

## Changelog

All notable changes to the project are documented here. This file is maintained automatically by [release-please](https://github.com/googleapis/release-please) from [Conventional Commit](https://www.conventionalcommits.org/) messages on `main`; release entries are prepended below as versions are tagged.
