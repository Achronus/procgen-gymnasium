## [unreleased]

### 💼 Other

- V0.1.1 release - Better build system (#3)

Github Actions build system is unreliable and hard to test. Refactored to a local approach for better usability.

### ⚙️ Miscellaneous Tasks

- *(test)* Added test verification for CI on OSs.
- *(publish)* Separated publish to PyPI as standalone workflow.

## [0.1.0] - 2026-02-10

### 💼 Other

- Merge for v0.1.0 release (#1)

* refactor(proj): Updated project folder name for packaging suitability.

* refactor(build): Removed redundant files.

* build(pkg): Updated project dependencies.

* feat(gym): Added gym3 connection interface to Gymnasium.

* refactor(gym): Refactored code from gym3 to use Gymnasium.

* revert(assets): Restored missing required data assets.

* fix(build): Updated CMake hardcoded values to auto-detection.

* fix(build): Fixed CMake build issues.

* docs(envs): Added documentation for each environment.

* feat(env): Added `gym.make()` support and metadata for wrappers.

* test(env): Added core unit tests and ones for official wrappers.

* docs(index): Updated README

* refactor(gym): Refactored package and docs to use `gym` convention.

* refactor(proj): Updated project name to `procgen_gym` for simplicity.

* build(ci): Add `git-cliff` for changelog management.

* ci(build): Added GitHub workflow for build and publishing.

* docs(env): Renamed environment list file.

* ci(fix): Fixed build CI workflow bug.
- Fix CI errors for v0.1.0 release (#2)

- Fix CMake build errors.
- Fix test build errors.
- Removed support for MacOS.
## [0.0.0] - 2026-02-08

### 💼 Other

- Initial commit
- Change copyright holder in LICENSE file

Updated copyright holder to Achronus.
- Added initial assets from `Procgen` original repo.

Repo: https://github.com/openai/procgen.
