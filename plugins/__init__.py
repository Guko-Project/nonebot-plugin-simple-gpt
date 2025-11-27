from __future__ import annotations

import importlib
import pkgutil

from nonebot.log import logger


def _load_builtin_plugins() -> None:
    for module_info in pkgutil.iter_modules(__path__):
        module_name = f"{__name__}.{module_info.name}"
        importlib.import_module(module_name)
        logger.debug("simple-gpt: 已加载插件模块 %s", module_name)


_load_builtin_plugins()
