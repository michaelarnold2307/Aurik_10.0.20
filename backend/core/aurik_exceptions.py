"""§v10.17 AurikException — spezifische Exception-Typen."""


class AurikException(Exception):  # noqa: N818 — stabiler öffentlicher API-Name seit v10
    """Basis für alle Aurik-Exceptions."""


class PhaseImportError(AurikException):
    """Phase-Modul konnte nicht importiert werden."""


class PhaseRuntimeError(AurikException):
    """Phase hatte einen Laufzeitfehler."""


class PhaseRegressionError(AurikException):
    """Phase verursacht Regression in Musical Goals."""


class PhasePSSRejection(AurikException):
    """Phase wurde vom PerceptualReferenceValidator verworfen."""


class PhaseTimeoutError(AurikException):
    """Phase überschritt das Zeitlimit."""


class PhaseOOMError(AurikException):
    """Phase verursachte Out-of-Memory."""


class PhaseShapeMismatch(AurikException):
    """Phase-Ausgabe hat falsche Shape."""


class CircuitBreakerTripped(AurikException):
    """Pipeline durch Circuit-Breaker abgebrochen."""


class ExportQualityFailure(AurikException):
    """Export-Qualitäts-Gate fehlgeschlagen."""
