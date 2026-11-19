"""
SafetyNet: Verhindert destruktive Änderungen durch Überwachung der Processing-Kette.
"""


class SafetyNet:
    """Prüft ob die Processing-Kette sicher (nicht-destruktiv) ist.

    Checks:
    - destructive-Flag in Output-Dict
    - clipping-Flag (|output| >> 1.0)
    - nan/inf-Flag
    - snr_degradation (falls vorhanden): SNR-Verschlechterung über Schwellwert
    """

    def __init__(
        self,
        max_snr_degradation_db: float = 20.0,
        clipping_tolerance: float = 0.01,
    ):
        self.max_snr_degradation_db = max_snr_degradation_db
        self.clipping_tolerance = clipping_tolerance

    def check_safety(self, chain, outputs: dict) -> bool:
        """Gibt True zurück wenn alle Module sicher abgeschlossen haben.

        :param chain: Liste der Module (unbenutzt, für Erweiterbarkeit)
        :param outputs: {module_name: output_dict}
        :return: bool (True = sicher)
        """
        for mod, out in outputs.items():
            if not isinstance(out, dict):
                continue
            # Direkte Destruktivitäts-Markierung
            if out.get("destructive", False):
                return False
            # NaN/Inf-Check
            if out.get("nan", False) or out.get("inf", False):
                return False
            # Clipping-Check
            clip_ratio = out.get("clipping_ratio", 0.0)
            if float(clip_ratio) > self.clipping_tolerance:
                return False
            # SNR-Degradations-Check
            snr_deg = out.get("snr_degradation_db", 0.0)
            if float(snr_deg) > self.max_snr_degradation_db:
                return False
        return True
