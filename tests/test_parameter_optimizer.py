# Test für KI-gestützte Parameteroptimierung

import numpy as np

from plugins.parameter_optimizer import ParameterOptimizer


def test_optimize():
    params = {"threshold": -20, "ratio": 2.0, "attack": 10}
    audio = np.random.randn(48000)
    targets = {"threshold": -18, "ratio": 3.0}
    optimizer = ParameterOptimizer()
    result = optimizer.optimize(params, audio, targets)
    assert result["threshold"] == -18
    assert result["ratio"] == 3.0
    print("Test Parameteroptimierung: OK")


if __name__ == "__main__":
    test_optimize()
