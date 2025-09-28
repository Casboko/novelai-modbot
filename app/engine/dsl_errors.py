from __future__ import annotations


class DslError(Exception):
    """Fatal DSL evaluation error."""


class DslValidationError(DslError):
    """Raised when a DSL expression fails validation."""


class DslRuntimeError(DslError):
    """Raised when runtime evaluation encounters an unrecoverable issue."""


class DslWarning(Warning):
    """Non-fatal DSL warning (logged but execution may continue)."""
