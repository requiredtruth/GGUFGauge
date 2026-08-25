#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ] || ! "$PY" -c "import ggufgauge, PySide6" >/dev/null 2>&1; then "$ROOT/install.sh"; fi
cd "$ROOT"
exec "$PY" -m ggufgauge.gui "$@"
