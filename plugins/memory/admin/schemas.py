from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PagingData(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


class ApiResponse(BaseModel):
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class HealthPayload(BaseModel):
    enabled: bool
    admin_enabled: bool
    debug_enabled: bool


class MetaPayload(BaseModel):
    scope: str
    top_k: int
    debug_max_rows: int
    routes_registered: bool
