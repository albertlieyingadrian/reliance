"""Put the repo root and this folder on the path, so `reliance` and `common` both import."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / 'tests'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: trains models; minutes rather than seconds')
