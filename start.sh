#!/bin/bash
# 启动 MiniMind 学习站:  ./start.sh [--port 8877]   然后打开 http://127.0.0.1:8877
cd "$(dirname "$0")"
PY=venv/bin/python; [ -x "$PY" ] || PY=python3
exec "$PY" server.py "$@"
