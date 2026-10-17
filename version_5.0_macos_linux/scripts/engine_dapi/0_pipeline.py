"""Compatibility entry point for the shared GUI pipeline (Stages 1–8).

The engine-local launcher used to duplicate labels, commands and completion checks.
Keep its familiar entry point, but open the same window as run_pharaoh_gui.sh.
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gui_pipeline as _shared

PipelineWindow = _shared.PipelineWindow


def __getattr__(name):
    return getattr(_shared, name)


if __name__ == "__main__":
    app = _shared.QApplication(sys.argv)
    window = PipelineWindow()
    window.show()
    sys.exit(app.exec_())
