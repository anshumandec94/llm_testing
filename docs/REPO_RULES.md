# Repository rules

- Prefer `uv` / `uv run` for repository commands.
- Do not modify `pyproject.toml` unless explicitly instructed.
- Prefer single config files as the source of truth for simulation runs instead of duplicating defaults in the CLI.
- Place generated analysis reports meant for later reading in `reports/` in the repository rather than session-state.
