#!/usr/bin/env python3
from pathlib import Path
import sys

# simple health check: ensure model file exists
if Path('model.joblib').exists():
    print('OK')
    sys.exit(0)
print('model missing')
sys.exit(1)
