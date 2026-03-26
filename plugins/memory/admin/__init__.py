from __future__ import annotations

from typing import Any


def register_admin_api(plugin: Any) -> bool:
    from .api import register_admin_api as _register_admin_api

    return _register_admin_api(plugin)


__all__ = ["register_admin_api"]
