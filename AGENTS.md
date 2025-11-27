# Repository Guidelines

<!-- 使用中文回答 -->
## Project Structure & Module Organization
The plugin lives in `src/plugins/nonebot-plugin-simple-gpt`, with `__init__.py` hosting the matcher wiring, history queue, and prompt builder, while `chat.py` wraps the shared `AsyncOpenAI` client. `pyproject.toml` defines the package metadata and uv-managed dependencies. Runtime configuration is injected through your host bot's `.env` or `.env.prod`; no assets or test fixtures are tracked yet, so add new resources under clearly named subfolders (for example, `tests/` or `assets/`) to keep the plugin directory tidy.

## Build, Test, and Development Commands
Use uv for reproducible environments:
```bash
uv sync                              # create/update virtualenv and install deps
uv run pip install -e .              # optional editable install when sharing the plugin
uv run nb run                        # start the host NoneBot instance to exercise the plugin
```
Automated tests are not shipped, but the repository expects future suites to run via `uv run pytest` from this folder.

## Coding Style & Naming Conventions
Stick to Python >=3.8, 4-space indentation, and type hints (the existing dataclasses, Pydantic config, and async helpers already rely on them). Prefer `snake_case` for functions and variables, `PascalCase` for classes such as `HistoryManager`, and uppercase constants for config defaults (see `IGNORED_PREFIXES`). Log through `nonebot.log.logger` instead of `print`, and keep prompt templates or strings localized near their consuming logic.

## Testing Guidelines
Add unit tests under `tests/`, mirroring module names (`tests/test_history_manager.py`, `tests/test_chat.py`). Favor `pytest` with `nonebot`'s testing utilities or `asyncio` fixtures to simulate message events. Cover edge cases like history trimming, proactive reply gating, and OpenAI failures. Run `uv run pytest` before every PR; shoot for meaningful assertions rather than a fixed coverage percentage.

## Commit & Pull Request Guidelines
Follow the short, lowercase Conventional Commit style already in history (`feat: ...`, `chore: ...`, `fix: ...`). Keep commits scoped to one concern and include context on how behavior changes for bot owners. PRs should describe the motivation, testing evidence (commands plus screenshots or logs when behavior changes), and link related issues. Call out new `.env` variables, migrations, or breaking changes explicitly so bot operators can react.

## Configuration & Security Notes
Environment keys beginning with `SIMPLE_GPT_` configure API credentials, prompt wording, and proactive reply rules. Never hardcode secrets; rely on `.env` and document defaults in `README.md`. When testing new API endpoints, prefer temporary keys and strip logs that contain prompts or user content before pushing.
