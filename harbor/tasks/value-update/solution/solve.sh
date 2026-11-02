#!/bin/sh

python - <<'PY'
from pathlib import Path

path = Path("/workspace/app.py")
path.write_text("VALUE = 2\n", encoding="utf-8")
PY
