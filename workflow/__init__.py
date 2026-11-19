"""
Workflow Management Package (GAP #26-28)

Unified workflow integration for AURIK combining:
- Batch Processing API (GAP #26)
- Undo/Redo System (GAP #27)
- Session Management Integration (GAP #28)

Author: AURIK Team
Version: 1.0.0
"""

from .workflow_manager import (  # Batch Processing; Undo/Redo; Session Management; Unified API
    BatchJobConfig,
    BatchJobResult,
    BatchProcessor,
    ProcessingState,
    UndoRedoManager,
    WorkflowManager,
    WorkflowSessionManager,
)

__all__ = [
    "BatchJobConfig",
    "BatchJobResult",
    "BatchProcessor",
    "ProcessingState",
    "UndoRedoManager",
    "WorkflowManager",
    "WorkflowSessionManager",
]
