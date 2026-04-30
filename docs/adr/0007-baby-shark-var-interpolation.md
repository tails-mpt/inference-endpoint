# ADR 0007: Add `${VAR}` interpolation to baby-shark TOML

**Status:** Accepted

## Context

The whole premise of this tool is "one tool, any model, any GPU shape." That
requires `accelerators` in `shark.toml` to be **dynamic per launch**, driven
by the `ACCELERATORS` `[[input]]` from slipstream's schema.

Investigation of baby-shark on `main` revealed:

- `[execution]` env vars reach the VM at runtime via the SkyPilot `envs:`
  block.
- **`[resources]` values are emitted verbatim** into the SkyPilot YAML
  (super_shark.py:382-386). No env var interpolation.
- Slipstream sets schema vars in `os.environ` before invoking shark, but
  shark never consults `os.environ` when reading TOML.

So `accelerators = "${ACCELERATORS}"` was parsed as the literal 17-char
string `"${ACCELERATORS}"`. SkyPilot rejects unknown GPU types, so launch
failed. Without a fix, every slipstream tool would have to fork itself
per GPU shape it ever wanted to support.

Workarounds considered:

| | Approach | Why rejected |
|---|---|---|
| Pre-stage script rewrites shark.toml | Hacky; extra plumbing; brittle in CI |
| Hardcode multiple shark.toml variants and switch | Not extensible; defeats the point |
| Promote to baby-shark itself | Right answer ↓ |

## Decision

Add bash-style interpolation to baby-shark's TOML readers as
[tails-mpt/baby-shark#110](https://github.com/tails-mpt/baby-shark/pull/110).

Syntax:
- `${VAR}` — required, fails fast if unset.
- `${VAR:-default}` — fallback when unset.
- `$$` — literal `$`.

Implemented in `super_shark.interpolate_env_vars()`, applied in
`read_shark_toml` and `read_shark_pipeline_toml`. Walks the parsed
manifest recursively, substitutes string values, raises `KeyError`
on missing required vars (returned as the standard `(None, error)`
tuple from the readers).

## Consequences

- **Cross-repo dependency.** This tool can't run cleanly until PR #110
  lands or users pin baby-shark to the feature branch. Worst-case:
  `accelerators` is a literal string and the launch fails fast, so no
  silent damage.
- **Benefits all future slipstream tools.** Resource fields (`cloud`,
  `disk_size`, `region`, `use_spot`, `ports`), the `[job]` block, and
  the `[stage.env]` block are all now parameterizable.
- **No syntax conflict with existing TOML.** `$$` is the standard escape;
  pre-existing `shark.toml` files don't contain `${...}` so there's no
  risk of breaking existing pipelines.
