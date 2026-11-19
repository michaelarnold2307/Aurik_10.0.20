"""
ONNX & Quantization Utility für Plugin-Integration

Dieses Modul stellt Hilfsfunktionen bereit, um ONNX-Modelle und Quantisierung einheitlich in allen Plugins zu nutzen.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_onnx_model(model_dir: Path, model_name: str) -> Path | None:
    """
    Prüft, ob ein ONNX-Modell im angegebenen Verzeichnis existiert.
    Gibt den Pfad zurück, falls vorhanden, sonst None.
    """
    onnx_path = model_dir / model_name
    if onnx_path.exists():
        logger.info("ONNX-Modell gefunden: %s", onnx_path)
        return onnx_path
    else:
        logger.warning("ONNX-Modell nicht gefunden: %s", onnx_path)
        return None


def quantize_onnx_model(onnx_path: Path, quantized_path: Path) -> bool:
    """
    Führt eine statische Quantisierung eines ONNX-Modells durch (sofern onnxruntime verfügbar).
    Speichert das quantisierte Modell unter quantized_path.
    Gibt True bei Erfolg, sonst False zurück.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_static

        # Dummy-Quantisierung (hier nur als Platzhalter, echte Kalibrierung je nach Modell nötig)
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            calibration_data_reader=None,  # Für echte Kalibrierung anpassen
            quant_format=QuantType.QOperator,
        )
        logger.info("ONNX-Modell quantisiert: %s", quantized_path)
        return True
    except ImportError:
        logger.error("onnxruntime nicht installiert. Quantisierung nicht möglich.")
        return False
    except Exception as e:
        logger.error("Quantisierung fehlgeschlagen: %s", e)
        return False
