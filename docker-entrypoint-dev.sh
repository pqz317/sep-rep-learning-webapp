#!/bin/sh
set -e
pip install --no-cache-dir -q -e .
exec "$@"
