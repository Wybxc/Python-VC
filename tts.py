import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from models.embed.inference import embed_utterance
from models.embed.inference import is_loaded as is_embed_loaded
from models.embed.inference import load_model as load_embed_model
from models.embed.inference import preprocess_wav
from models.hifigan.inference import infer_waveform
from models.hifigan.inference import is_loaded as is_hifigan_loaded
from models.hifigan.inference import load_model as load_hifigan_model
from models.tacotron.synthesizer import Synthesizer

_synthesizer: Optional[Synthesizer] = None


def load_models(
    synthesizer_path: Union[str, Path],
    embed_path: Union[str, Path],
    hifigan_path: Union[str, Path],
    hifigan_config_path: Union[str, Path],
):
    """加载模型。"""
    global _synthesizer
    _synthesizer = Synthesizer(Path(synthesizer_path))
    load_embed_model(Path(embed_path))
    load_hifigan_model(Path(hifigan_path), Path(hifigan_config_path))


def speaker_embed(wav: Union[str, Path, np.ndarray]):
    """说话人风格嵌入。"""
    if not is_embed_loaded():
        raise RuntimeError("请先初始化 Embed 模型。")
    wav = preprocess_wav(wav)
    return embed_utterance(wav)


def synthesize(
    sentence: str,
    embed: np.ndarray,
    style_idx: int = -1,
    accurancy: int = 5,
    max_sentence_length: int = 2,
):
    """Tacotron 文本转语谱。

    Args:
        sentence (str): 要转换的文本。
        style_idx (int, optional): 样式索引。 Defaults to -1.
        min_stop_token (int, optional): 精度。 Defaults to 5.
        max_sentence_length (int, optional): 最大句子长度。 Defaults to 2.
    """
    global _synthesizer
    if _synthesizer is None:
        raise RuntimeError("请先初始化 Tacotron 模型。")
    # 文本预处理
    text_splited = sum((re.split(r"[！，。、,]", s) for s in sentence.split("\n")), [])
    text_splited = [str(s) for s in text_splited if s]
    # 说话人风格嵌入
    speaker_embeds = [embed] * len(text_splited)
    # 分段生成语谱
    spectrograms = _synthesizer.synthesize_spectrograms(
        text_splited,
        speaker_embeds,
        style_idx=style_idx,
        min_stop_token=accurancy,
        steps=max_sentence_length * 200,
    )
    # 拼接语谱
    breaks = [spec.shape[1] for spec in spectrograms]  # type: ignore
    spectrogram = np.concatenate(spectrograms, axis=1)
    return spectrogram, breaks


def vocode(spectrogram: np.ndarray, breaks: Sequence[int]):
    """声码器，将语谱转换为声音。"""
    if not is_hifigan_loaded():
        raise RuntimeError("请先初始化 hifi-gan 模型。")
    # 生成语音
    wav, sample_rate = infer_waveform(spectrogram)
    # 拆分语音
    break_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)  # type: ignore
    break_starts = np.concatenate((np.array([0]), break_ends[:-1]))
    wav_slices = [wav[start:end] for start, end, in zip(break_starts, break_ends)]
    break_slices = [np.zeros(int(0.15 * sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wav_slices, break_slices) for i in (w, b)])
    # 清除多余空白
    wav = preprocess_wav(wav)
    # Play it
    wav = wav / np.abs(wav).max() * 0.97
    return wav, sample_rate
