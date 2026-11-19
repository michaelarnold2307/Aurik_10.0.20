"""Example Plugin — Minimalbeispiel für Aurik Plugin-SDK.

§15.6: Kopiere dieses Verzeichnis als Startpunkt für dein eigenes Plugin.

    cp -r plugins/sdk/example_plugin plugins/mein_plugin
    cd plugins/mein_plugin
    # Bearbeite mein_plugin.py und manifest.json
    pytest test_example.py -v
"""

from plugins.sdk.aurik_plugin_base import AurikPlugin, PluginManifest


class ExamplePlugin(AurikPlugin):
    """Minimalbeispiel: Gain-Anpassung."""

    manifest = PluginManifest(
        name="example-plugin",
        version="1.0.0",
        description="Beispiel-Plugin: Gain-Anpassung mit dB-Steuerung",
        author="Aurik SDK",
        license="MIT",
        min_aurik_version="10.0.0",
        dependencies=["numpy>=1.21"],
        tags=["example", "gain", "tutorial"],
    )

    def process_audio(self, audio, sr=48000, gain_db=0.0, **kwargs):
        """Wendet Gain in dB auf das Audio an.

        Args:
            audio:   Eingabe-Audio (float32).
            sr:      Abtastrate.
            gain_db: Gain in dB (positiv = lauter, negativ = leiser).
            **kwargs: Weitere Parameter (ignoriert).

        Returns:
            Audio mit Gain-Anpassung.
        """
        import numpy as np

        gain_linear = 10 ** (gain_db / 20)
        return (audio * gain_linear).astype(np.float32)

    def validate(self):
        """Selbsttest."""
        ok, msg = super().validate()
        if not ok:
            return ok, msg
        import numpy as np

        try:
            test = np.zeros(100, dtype=np.float32)
            result = self.process_audio(test, sr=48000, gain_db=0.0)
            assert result.shape == test.shape, f"Shape mismatch: {result.shape} != {test.shape}"
            assert np.all(np.isfinite(result)), "Non-finite values in output"
            return True, "OK"
        except Exception as exc:
            return False, f"Self-test failed: {exc}"
