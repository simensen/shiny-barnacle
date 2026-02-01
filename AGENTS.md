# AGENTS.md

LLM Response Transformer Proxy. Python 3.12+, FastAPI, async throughout.

## Critical Rules

**Environment:**
- ALWAYS run commands from the project directory
- ALWAYS try commands failing because "command not found" again with the `./scripts/run.sh` (example: if `make check` fails because `python3` or the `uv` command was not found, try the command again as `./scripts/run.sh make check`)
- NEVER modify Makefiles, scripts, or configs to fix "command not found" errors
- NEVER try `python3`, `/usr/bin/python`, or search for interpreters — fix your environment

**Workflow:**
- ALWAYS use `make` commands, not direct tool invocation
- ALWAYS run `make check` before considering any code task complete

## Commands
```bash
make dev       # Install dev dependencies
make test      # Run tests
make check     # lint + typecheck + test (REQUIRED before completing code tasks)
make format    # Format code
```

## Environment Setup

This project uses nix-direnv.

**Preferred:** From the project directory:
```bash
make test             # direnv activates automatically
```

**If direnv reports "blocked":** Run `direnv allow` once, then commands work normally:
```bash
direnv allow
make test
```

**From anywhere, or if each command runs in a fresh shell:** Prefix commands with `direnv exec /path/to/project`:
```bash
direnv exec /path/to/project make test
```

**To check:** Run `direnv status` from the project directory.

## If: "command not found" (python, uv, make)

Your environment is not loaded. The project is configured correctly.

**Do:** Run `direnv allow` from the project directory, or use `direnv exec /path/to/project <command>`.

**Don't:**
- Edit the Makefile to use `python3`
- Search for Python elsewhere
- Install dependencies manually
- Modify any project configuration

## If: Running Tests, Linting, or Type Checking

**Do:** Use make targets.
```bash
make test      # not: pytest
make format    # not: ruff format
make check     # not: ruff check && mypy && pytest
```

**Don't:** Invoke pytest, ruff, or mypy directly.

## If: You Changed Code

Run `make check` and confirm it passes before marking the task complete.

This is required for every code change.

## If: Writing Tests

See [docs/testing-patterns.md](docs/testing-patterns.md) for detailed patterns.

## Code Style

- Line length: 100
- Type hints required (mypy strict)
- Ruff rules: E, F, I, N, W, UP, B, C4, SIM
