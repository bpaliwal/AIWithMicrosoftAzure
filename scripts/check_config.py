#!/usr/bin/env python3
"""Quick CLI to validate configuration and probe configured endpoints.

Usage:
    python scripts/check_config.py
"""
import sys
from src.config import Config


def main() -> int:
    print("Validating required environment variables...")
    try:
        Config.validate()
        print("  ✔ Required env vars present")
    except Exception as e:
        print("  ✖ Validation failed:", e)
        return 2

    print("\nProbing connections (short timeouts)...")
    results = Config.validate_connections(timeout=3, parallel=True)
    ok = True
    for name, res in results.items():
        status = "OK" if res.get("ok") else "FAIL"
        code = res.get("status_code")
        print(f" - {name}: {status} ({res.get('msg')}) status_code={code}")
        if not res.get("ok"):
            ok = False

    if not ok:
        print("\nSome probes failed. Fix environment or run with more diagnostics.")
        return 3

    print("\nAll probes passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
