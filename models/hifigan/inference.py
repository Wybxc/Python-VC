from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
import pathlib
from typing import Tuple

import numpy as np
import torch

from .env import AttrDict
from .models import Generator

generator = None  # type: Generator | None
output_sample_rate = None  # type: int | None
_device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_model(weights_fpath, config_fpath=None, verbose=True):
    global generator, _device, output_sample_rate

    if verbose:
        print("Building hifigan")

    if config_fpath is None:
        config_fpath = "./models/hifigan/config_16k_.json"
    data = pathlib.Path(config_fpath).read_text()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    output_sample_rate = h.sampling_rate
    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        # _model = _model.cuda()
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    generator = Generator(h).to(_device)
    state_dict_g = load_checkpoint(weights_fpath, _device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()


def vocoder_config():
    if generator is None:
        raise RuntimeError("Please load hifi-gan in memory before using it")
    return generator.h


def is_loaded():
    return generator is not None


def infer_waveform(mel, progress_callback=None) -> Tuple[np.ndarray, int]:

    if generator is None:
        raise RuntimeError("Please load hifi-gan in memory before using it")

    mel = torch.FloatTensor(mel).to(_device)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
    audio = audio.cpu().numpy()

    return audio, output_sample_rate or 16000
