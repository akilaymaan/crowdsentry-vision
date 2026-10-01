#!/usr/bin/env sh
set -e

# Mongo needs no schema migration step: app startup runs ensure_indexes(), which is
# idempotent, so every container start converges the indexes before accepting traffic.
# On a multi-replica deployment this is still safe -- createIndex is a no-op when the
# index already exists.

# Exactly one worker process: the API embeds the camera workers, so --workers >1 would
# duplicate every feed's processing and double-write observations.
exec uvicorn app.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --workers 1
