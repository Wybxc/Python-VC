from pathlib import Path
from typing import Union

import numpy as np
import sounddevice as sd
import soundfile as sf


def play(wav: np.ndarray, sample_rate: int):
    """播放音频。"""
    try:
        sd.stop()
        sd.play(wav, sample_rate)
    except Exception as e:
        print(e)
        print("Error in audio playback. Try selecting a different audio output device.")
        print("Your device must be connected before you start the toolbox.")


def save(path: Union[str, Path], wav: np.ndarray, sample_rate: int):
    """保存音频。"""
    if Path(path).suffix == "":
        path = Path(path).with_suffix(".wav")
    sf.write(path, wav, sample_rate)
