"""IPC-Modul für Aurik — geteilte Speicher und Pipeline-Prozess-Kommunikation.

Architektur:
  - SharedAudioRing: Lock-freier Ringpuffer für Audio-Frames via multiprocessing.shared_memory
  - PipelineProcess: Eigenständiger Prozess für UV3-Restaurierung
  - Polling-Timer im GUI-Thread liest neue Frames ohne Eventloop-Blockade
"""

from Aurik10.ipc.shared_audio import AudioFrame, RingInfo, SharedAudioRing

__all__ = [
    "SharedAudioRing",
    "AudioFrame",
    "RingInfo",
]
