"""
AURIK v8 Conduct Enforcer Package
==================================

Nicht umgehbare Enforcement-Layer für alle Processing-Operationen.
Validiert Musical Goals, Conduct Principles, und v8 Architecture Rules.

Classes:
    ConductEnforcer: Main enforcer class
    ValidationResult: Result of validation
    Zone: Processing zones (A/B/C)

Example:
    >>> from backend.core.conduct_enforcer import ConductEnforcer
    >>> enforcer = ConductEnforcer()
    >>> result = enforcer.validate_step(
    ...     cas_delta=0.020,
    ...     dcs=0.12,
    ...     listener_diff=0.25,
    ...     uncertainty=0.15,
    ...     irreversible=False,
    ...     musical_goals_pre={'brillanz': 0.82, ...},
    ...     musical_goals_predicted={'brillanz': 0.87, ...}
    ... )
    >>> if result.allowed:
    ...     # Proceed with processing
    ...     pass
    ... else:
    ...     # Hard Stop!
    logger.debug("Blocked: %s", result.reason)
"""

import logging

from .conduct_enforcer import ConductEnforcer, ValidationResult, Zone

logger = logging.getLogger(__name__)

__all__ = ["ConductEnforcer", "ValidationResult", "Zone"]
__version__ = "9.15.0"
