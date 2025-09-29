from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from .engine.loader import load_rules_result
from .engine.types import DslMode, DslPolicy, LoadResult

logger = logging.getLogger(__name__)

_INVALID_ENV_LOGGED = False


def _normalize_env_mode(raw: Optional[str]) -> Optional[DslMode]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"warn", "strict"}:
        return value  # type: ignore[return-value]
    global _INVALID_ENV_LOGGED
    if not _INVALID_ENV_LOGGED:
        logger.warning(
            "Ignoring MODBOT_DSL_MODE=%r (expected 'warn' or 'strict')",
            raw,
        )
        _INVALID_ENV_LOGGED = True
    return None


def resolve_policy(
    rules_path: Path | str,
    cli_mode: Optional[str],
) -> Tuple[DslPolicy, LoadResult, Optional[DslMode]]:
    """Resolve DSL policy using CLI/ENV/YAML precedence.

    Returns the effective policy, loader result snapshot, and the normalized
    environment mode (if valid).
    """
    env_raw = os.getenv("MODBOT_DSL_MODE")
    env_mode = _normalize_env_mode(env_raw)
    override = cli_mode or env_mode
    result = load_rules_result(rules_path, override_mode=override)
    policy = DslPolicy.from_mode(result.mode)
    return policy, result, env_mode


def has_version_mismatch(result: LoadResult) -> bool:
    """Return True when the loader reported a rules version mismatch."""
    for issue in result.issues:
        if issue.code == "R2-V001" and issue.where == "version":
            return True
    return False
