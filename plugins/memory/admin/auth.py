from __future__ import annotations

from fastapi import Header, HTTPException, status


def verify_admin_token(
    authorization: str = Header(default="", alias="Authorization"),
) -> None:
    from .. import _get_plugin_config

    config = _get_plugin_config()
    token = config.simple_gpt_memory_admin_token.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="memory admin token not configured",
        )

    expected = f"Bearer {token}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin token",
        )
