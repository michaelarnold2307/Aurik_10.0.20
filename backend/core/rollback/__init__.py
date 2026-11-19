"""
AURIK v8 Rollback Package
==========================

Snapshot management and rollback functionality for audio processing pipeline.

Components:
- RollbackManager: Create snapshots and rollback functionality
- AudioSnapshot: Snapshot data structure
- RollbackDecision: Rollback decision audit trail

Version: 8.0.0
"""

from .rollback_manager import AudioSnapshot, RollbackDecision, RollbackManager

__all__ = ["AudioSnapshot", "RollbackDecision", "RollbackManager"]

__version__ = "9.15.0"
