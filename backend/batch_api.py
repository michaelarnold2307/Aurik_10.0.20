"""
Rückwärtskompatibilitäts-Shim (§9.4 Anti-Parallelwelten-Prinzip).

Die kanonische Implementierung liegt in ``backend/api/rest/batch_api.py``.
Lazy-Import vermeidet Initialisierungsprobleme des backend.api-Pakets.
"""


def __getattr__(name: str):
    from backend.api.rest import batch_api as _m

    return getattr(_m, name)
