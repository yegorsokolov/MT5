#!/usr/bin/env bash
set -e

# Detect the resource tier and load any plugins before executing the requested
# command.  This will print the tier so users can verify which capabilities were
# detected.
python -m utils.resource_monitor

exec "$@"
