import logging

import requests

logger = logging.getLogger(__name__)


def dccrn_infer_wav(wav_path: str, api_url: str = "http://localhost:8501/infer/") -> str | None:
    """Infer DCCRN denoising via REST API.

    Args:
        wav_path: Path to input WAV file
        api_url: DCCRN API endpoint URL

    Returns:
        Path to output file if successful, None otherwise
    """
    with open(wav_path, "rb") as f:
        files = {"audio": ("input.wav", f, "audio/wav")}
        r = requests.post(api_url, files=files)
        r.raise_for_status()
        data = r.json()
        # Ergebnis als WAV speichern (falls benötigt)
        if "result_wav" in data:
            out_path = wav_path.replace(".wav", "_dccrn.wav")
            with open(out_path, "wb") as out:
                out.write(bytes.fromhex(data["result_wav"]))
            logger.info("DCCRN-Ergebnis gespeichert: %s", out_path)
            return out_path
        else:
            logger.warning("Kein Ergebnis erhalten: %s", data)
            return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logger.error("Nutzung: python dccrn_client.py <eingabe.wav>")
        exit(1)
    dccrn_infer_wav(sys.argv[1])
