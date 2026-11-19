import json
import os


def test_peer_review_feedback(tmp_path):
    # Simuliere Peer-Review-Feedback
    feedback = {
        "example": [
            {
                "step": "spectral_repair",
                "reviewer": "experte@aurik.com",
                "feedback": "Natürlichkeit erhalten",
                "rating": 5,
            }
        ]
    }
    feedback_path = tmp_path / "peer_review_feedback.json"
    os.makedirs(tmp_path, exist_ok=True)
    with open(feedback_path, "w") as f:
        json.dump(feedback, f)
    # Prüfe, ob Feedback für spectral_repair gefunden wird
    with open(feedback_path) as f:
        data = json.load(f)
    relevant = [entry for entry in data.get("example", []) if entry.get("step") == "spectral_repair"]
    assert relevant, "Kein Peer-Review-Feedback für spectral_repair gefunden!"
    import logging

    logging.info("Peer-Review-Feedback Test bestanden.")
