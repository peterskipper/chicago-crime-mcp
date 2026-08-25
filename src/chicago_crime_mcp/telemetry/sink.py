"""Where the records go: newline-delimited JSON, one file per day.

**Why files and not a table.** The obvious alternative was writing straight into
DuckDB, and JSONL wins on three counts. It is what a deployed log drain produces
anyway, so the same rollup query runs over local files and over shipped logs
without a second code path. DuckDB reads it natively, so the batch job is a
query rather than a parser. And retention -- a real obligation in an
investigative domain, not a nicety -- is ``rm``, rather than a ``DELETE`` that
has to contend with whatever else holds the database open.

**Why one file per day.** It makes retention a file operation and it bounds how
much has to be re-read to recompute a window. The rollup globs them.

**Telemetry must never be the reason a tool call fails.** Every write is guarded,
and the first failure disables the sink for the life of the process after
logging once. A full disk should cost observability, not the server. The stderr
mirror keeps working either way, since that is the logging module's problem
rather than this one's.

Writes are synchronous. Appending a line to an open handle and flushing it is
microseconds, which is not worth a queue and a drain task on the event loop --
and a queue would introduce the one failure mode this design is trying to avoid,
where the records that matter most are the ones lost on the way down.

Docstrings follow the Google Python style.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from chicago_crime_mcp.telemetry.record import CallRecord

log = logging.getLogger(__name__)

#: Default location, alongside the other things the pipeline writes.
DEFAULT_LOG_DIR = Path("data/telemetry")

#: Filename pattern. The date is the record's own, not the file's creation time,
#: so a call at 23:59:59.9 lands in the day it belongs to.
FILENAME = "calls-{day}.jsonl"


@dataclass(frozen=True)
class TelemetryConfig:
    """Where to write, and whether to.

    Follows :class:`~chicago_crime_mcp.store.config.StoreConfig`: environment
    variables only, defaults that work in a fresh checkout with no ``.env``.

    Attributes:
        enabled: Whether records are written to disk at all. The stderr mirror
            is independent of this and is controlled by the logging level.
        log_dir: Directory holding the daily files. Created on first write.
    """

    enabled: bool = True
    log_dir: Path = DEFAULT_LOG_DIR

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TelemetryConfig:
        """Build a config from environment variables.

        Args:
            environ: Environment mapping to read (defaults to ``os.environ``).
                Injectable so tests need not mutate process globals.

        Returns:
            A populated :class:`TelemetryConfig`.
        """
        env = os.environ if environ is None else environ
        raw = env.get("TELEMETRY_ENABLED", "1").strip().lower()
        return cls(
            enabled=raw not in {"0", "false", "no", "off"},
            log_dir=Path(env.get("TELEMETRY_LOG_DIR", str(DEFAULT_LOG_DIR))),
        )


class TelemetrySink:
    """An append-only JSONL writer that rotates daily and never raises.

    The handle stays open between writes and is flushed after each one, so a
    rollup can read a file the server is still writing to.
    """

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        """Create a sink.

        Nothing is opened here -- the first :meth:`write` creates the directory
        and the file. A server that is never called leaves no empty artifacts,
        and a bad path is reported when a record is actually lost rather than at
        import time.

        Args:
            config: Where to write. Defaults to the environment's.
        """
        self.config = config or TelemetryConfig.from_env()
        self._lock = threading.Lock()
        self._handle: IO[str] | None = None
        self._day: str | None = None
        self._broken = False
        atexit.register(self.close)

    def write(self, record: CallRecord) -> None:
        """Append one record, and mirror a one-line summary to the log.

        The mirror happens even when the file sink is disabled or broken: a
        local stdio session usually wants the line on stderr and nothing on
        disk, and a sink that has failed should still leave a trace of the calls
        it could not record.

        Args:
            record: The call to log.
        """
        log.info("%s", record.summary())
        if not self.config.enabled or self._broken:
            return
        line = json.dumps(record.to_dict(), default=str)
        day = record.ts[:10]
        try:
            with self._lock:
                handle = self._rotate_to(day)
                handle.write(line + "\n")
                handle.flush()
        except OSError as exc:
            # Once, then never again: a broken sink that logs per call would
            # bury the tool errors this exists to surface.
            self._broken = True
            log.warning("telemetry sink disabled after write failure: %s", exc)

    def close(self) -> None:
        """Close the open handle, if any. Safe to call more than once."""
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:  # pragma: no cover - nothing useful to do
                    pass
                self._handle = None
                self._day = None

    def path_for(self, day: str) -> Path:
        """Return the file a record from ``day`` belongs in.

        Args:
            day: An ISO date, ``YYYY-MM-DD``.

        Returns:
            The full path, which may not exist yet.
        """
        return self.config.log_dir / FILENAME.format(day=day)

    def _rotate_to(self, day: str) -> IO[str]:
        """Return the open handle for ``day``, rotating to it if needed.

        Returns the handle rather than only assigning it so the caller does not
        have to re-narrow ``self._handle`` from ``IO[str] | None``.

        Args:
            day: An ISO date, ``YYYY-MM-DD``.

        Returns:
            The open, appendable handle for that day's file.

        Raises:
            OSError: If the directory or file cannot be opened. Caught by
                :meth:`write`, which is the only caller.
        """
        if self._handle is not None and self._day == day:
            return self._handle
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._handle = self.path_for(day).open("a", encoding="utf-8")
        self._day = day
        return self._handle


__all__ = ["DEFAULT_LOG_DIR", "FILENAME", "TelemetryConfig", "TelemetrySink"]
