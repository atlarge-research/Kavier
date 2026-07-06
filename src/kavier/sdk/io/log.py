"""Shared Rich console logger; exports the timestamped ``log`` callable."""

from rich.console import Console

_log = Console()
log = _log.log

__all__ = ["log"]
