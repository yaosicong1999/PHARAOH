"""Run PHARAOH entry points in Python's UTF-8 mode on Windows.

On macOS and Linux text I/O is UTF-8 already. Windows Python defaults to the ANSI
code page (cp1252) for open() and for redirected stdout, and the pipeline both prints
and writes non-ASCII text -- µm, ±, →, ✅, and JSON dumped with ensure_ascii=False --
which then fails to encode or reads back as mojibake. UTF-8 mode (PEP 540) makes every
default text encoding UTF-8, the same as on a Mac, without touching each open().

`ensure_utf8_mode()` is called first thing by each entry point. Elsewhere, or when
the interpreter is already in UTF-8 mode (PYTHONUTF8=1, `python -X utf8`, or a child
the parallel driver started), it does nothing. Otherwise it re-runs the same command
under `-X utf8` and exits with that run's exit code; PYTHONUTF8=1 is exported too, so
every Python the run starts inherits the mode.
"""

import os
import subprocess
import sys


def ensure_utf8_mode():
    if sys.platform != "win32" or sys.flags.utf8_mode:
        return
    env = dict(os.environ, PYTHONUTF8="1")
    child = subprocess.Popen([sys.executable, "-X", "utf8", *sys.argv], env=env)
    while True:
        try:
            raise SystemExit(child.wait())
        except KeyboardInterrupt:
            # The console sends Ctrl+C to the child too; keep waiting while it stops
            # its own work, then exit with its code.
            continue
