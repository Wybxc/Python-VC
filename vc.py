from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from models.convertor import MelDecoderMOLv2
from models.convertor import load_model as load_convertor_model
from models.embed.inference import embed_utterance
from models.embed.inference import is_loaded as is_embed_loaded
from models.embed.inference import load_model as load_embed_model
from models.embed.inference import preprocess_wav
from models.extractor import PPGModel
from models.extractor import load_model as load_extractor_model
from models.hifigan.inference import infer_waveform
from models.hifigan.inference import is_loaded as is_hifigan_loaded
from models.hifigan.inference import load_model as load_hifigan_model
from models.hifigan.inference import vocoder_config
from utils.f0_utils import compute_f0, compute_mean_std, f02lf0, get_converted_lf0uv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_extractor: Optional[PPGModel] = None
_convertor: Optional[MelDecoderMOLv2] = None


def load_models(
    extractor_path: Union[str, Path],
    convertor_config_path: Union[str, Path],
    convertor_path: Union[str, Path],
    embed_path: Union[str, Path],
    hifigan_path: Union[str, Path],
    hifigan_config_path: Union[str, Path],
):
    global _extractor, _convertor
    _extractor = load_extractor_model(Path(extractor_path))
    _convertor = load_convertor_model(Path(convertor_config_path), Path(convertor_path))
    load_embed_model(Path(embed_path))
    load_hifigan_model(Path(hifigan_path), Path(hifigan_config_path))


def is_model_loaded():
    return _extractor is not None and _convertor is not None

class VoiceTarget:
    wav: np.ndarray
    embed: np.ndarray

    def __init__(self, wav: Union[str, Path, np.ndarray]):
        self.wav, self.embed = speaker_embed(wav)


def speaker_embed(wav: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """说话人风格嵌入。"""
    if not is_embed_loaded():
        raise RuntimeError("请先初始化 Embed 模型。")
    wav_preprocessed = preprocess_wav(wav)
    return wav_preprocessed, embed_utterance(wav_preprocessed)


def convert(source_wav: Union[str, Path, np.ndarray], target: VoiceTarget):
    """将音频转换为 PPG 信号。"""
    if _extractor is None or _convertor is None:
        raise RuntimeError("请先初始化 PPG 模型。")
    # 加载音频
    wav = preprocess_wav(source_wav)
    # 提取 PPG
    ppg = _extractor.extract_from_wav(wav)
    # 计算目标音频的 F0
    ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(target.wav)))
    lf0_uv = get_converted_lf0uv(wav, ref_lf0_mean, ref_lf0_std, convert=True)
    # 裁剪
    min_len = min(ppg.shape[1], len(lf0_uv))
    ppg = ppg[:, :min_len]
    lf0_uv = lf0_uv[:min_len]
    _, mel_pred, _ = _convertor.inference(
        ppg,
        logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device),
        spembs=torch.from_numpy(target.embed).unsqueeze(0).to(device),
    )
    mel_pred = mel_pred.transpose(0, 1)
    breaks = [mel_pred.shape[1]]
    mel_pred = mel_pred.detach().cpu().numpy()
    return mel_pred, breaks


def vocode(spectrogram: np.ndarray, breaks: Sequence[int]):
    """声码器，将语谱转换为声音。"""
    if not is_hifigan_loaded():
        raise RuntimeError("请先初始化 hifi-gan 模型。")
    # 生成语音
    wav, sample_rate = infer_waveform(spectrogram)
    # 拆分语音
    break_ends = np.cumsum(np.array(breaks) * vocoder_config().hop_size)  # type: ignore
    break_starts = np.concatenate((np.array([0]), break_ends[:-1]))
    wav_slices = [wav[start:end] for start, end, in zip(break_starts, break_ends)]
    break_slices = [np.zeros(int(0.15 * sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wav_slices, break_slices) for i in (w, b)])
    # 清除多余空白
    wav = preprocess_wav(wav)
    # Play it
    wav = wav / np.abs(wav).max() * 0.97
    return wav, sample_rate
