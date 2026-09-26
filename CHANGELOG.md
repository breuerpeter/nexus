# Changelog

## [0.2.0](https://github.com/breuerpeter/nexus/compare/nexus-sim-v0.1.0...nexus-sim-v0.2.0) (2026-09-26)


### ⚠ BREAKING CHANGES

* **cli:** --runtime leaves the command-line tool, LaunchConfig's runtime.backend and the empty isaacsim extra go, the isaacsim compose service and its image go, and nexus script runs only Kit-only asset scripts in the Kit image, with no nexus code inside.
* **config:** ship two Astro Max vehicles in the catalog ([#4](https://github.com/breuerpeter/nexus/issues/4))

### Features

* **config:** a project catalog extends the bundled one ([#17](https://github.com/breuerpeter/nexus/issues/17)) ([b520f74](https://github.com/breuerpeter/nexus/commit/b520f7406e2f715e7f8bfce6ce3959fc2aa0a2d4))
* **config:** ship two Astro Max vehicles in the catalog ([#4](https://github.com/breuerpeter/nexus/issues/4)) ([32e1b2f](https://github.com/breuerpeter/nexus/commit/32e1b2f93aa9c2cbcf09ef7c2edd72f7399a0cab)), closes [#3](https://github.com/breuerpeter/nexus/issues/3)


### Bug Fixes

* **ci:** gate the rl leg by the kind of quantity each check reads ([#68](https://github.com/breuerpeter/nexus/issues/68)) ([ab8dc8a](https://github.com/breuerpeter/nexus/commit/ab8dc8a42544f29c63ab5f9e5615cdbc73510449)), closes [#15](https://github.com/breuerpeter/nexus/issues/15)
* **ci:** let an approved fork run check out the fork's code ([#24](https://github.com/breuerpeter/nexus/issues/24)) ([7ab3a28](https://github.com/breuerpeter/nexus/commit/7ab3a2810ea317d87ddf729cac5173ae456a53f3))
* **cli:** launch the Kit container from an installed package ([#22](https://github.com/breuerpeter/nexus/issues/22)) ([821a007](https://github.com/breuerpeter/nexus/commit/821a0073bd096671066e3311de3054b19cf09cbe)), closes [#12](https://github.com/breuerpeter/nexus/issues/12)
* **examples:** goto_policy fetches its hosted policy from the catalog base ([#67](https://github.com/breuerpeter/nexus/issues/67)) ([1445f6a](https://github.com/breuerpeter/nexus/commit/1445f6a460123da7bde2ef1f75f31324e2818819)), closes [#66](https://github.com/breuerpeter/nexus/issues/66)
* **orchestrator:** capture the host-exchange graph before PX4 dials in ([#61](https://github.com/breuerpeter/nexus/issues/61)) ([94009bc](https://github.com/breuerpeter/nexus/commit/94009bc1ba3c738f720bf4e7dbec7b840eda0d88)), closes [#26](https://github.com/breuerpeter/nexus/issues/26)
* **px4:** a late PX4 dial-in logs no ekf2 missing data line ([#60](https://github.com/breuerpeter/nexus/issues/60)) ([cab8ef9](https://github.com/breuerpeter/nexus/commit/cab8ef92bd31469d5666679aa6417a3322b3ee69)), closes [#57](https://github.com/breuerpeter/nexus/issues/57)
* **px4:** allowlist the missing-parameter message, not [param] ([#23](https://github.com/breuerpeter/nexus/issues/23)) ([eb78bb9](https://github.com/breuerpeter/nexus/commit/eb78bb9a9e9f8a973a8a28c62b4e1d9caa2e6e74)), closes [#21](https://github.com/breuerpeter/nexus/issues/21)

## Changelog

All notable changes to the project are documented here. This file is maintained automatically by [release-please](https://github.com/googleapis/release-please) from [Conventional Commit](https://www.conventionalcommits.org/) messages on `main`; release entries are prepended below as versions are tagged.
