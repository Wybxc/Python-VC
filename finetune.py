from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import numpy as np
import torch

from models.embed.inference import embed_utterance
from models.embed.inference import is_loaded as is_embed_loaded
from models.convertor import MelDecoderMOLv2
from models.convertor import load_model as load_convertor_model
from models.convertor.loss import MaskedMSELoss
from models.embed.audio import preprocess_wav
from models.extractor import PPGModel
from models.extractor import load_model as load_extractor_model
from torch.utils.data import DataLoader, Dataset
from utils.f0_utils import get_converted_lf0uv

from utils.signals import mel_spectrogram


def load_models(
    extractor_path: Union[str, Path],
    convertor_config_path: Union[str, Path],
    convertor_path: Union[str, Path],
):
    extractor = load_extractor_model(Path(extractor_path))
    convertor = load_convertor_model(Path(convertor_config_path), Path(convertor_path))
    return extractor, convertor


def speaker_embed(wav: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """说话人风格嵌入。"""
    if not is_embed_loaded():
        raise RuntimeError("请先初始化 Embed 模型。")
    wav_preprocessed = preprocess_wav(wav)
    return wav_preprocessed, embed_utterance(wav_preprocessed)


class VCFinetune(pl.LightningModule):
    def __init__(self, extractor: PPGModel, convertor: MelDecoderMOLv2):
        super().__init__()
        self.extractor = extractor
        self.convertor = convertor
        self.loss_criterion = MaskedMSELoss(convertor.frames_per_step)

    def forward(self, wav, in_lengths, mels, out_lengths, lf0_uvs, speaker_embeds):
        ppg = self.extractor(wav, in_lengths)
        il = min(ppg.shape[1], lf0_uvs.shape[1])
        mel_outputs, mel_outputs_postnet, predicted_stop = self.convertor(
            ppg[:, :il, :],
            in_lengths.div(160, rounding_mode='trunc'),
            mels,
            out_lengths,
            lf0_uvs[:, :il, :],
            speaker_embeds,
        )
        return mel_outputs, mel_outputs_postnet, predicted_stop

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-6)

    def training_step(self, batch, batch_idx):
        wav, in_lengths, mels, out_lengths, lf0_uvs, speaker_embeds, stop_tokens = batch
        mel_outputs, mel_outputs_postnet, predicted_stop = self(
            wav, in_lengths, mels, out_lengths, lf0_uvs, speaker_embeds
        )
        mel_loss, stop_loss = self.loss_criterion(
            mel_outputs,
            mel_outputs_postnet,
            mels,
            out_lengths,
            stop_tokens,
            predicted_stop,
        )
        loss = mel_loss
        return {"loss": loss}


class AiTangDataset(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        embeds_path: Union[str, Path],
        lf0_path: Union[str, Path],
    ):
        self.root_dir = Path(root_dir)
        self.filenames = sorted(self.root_dir.rglob("*.wav"))
        self.size = len(self.filenames)

        self.embeds = np.load(Path(embeds_path))
        self.lf0 = np.load(Path(lf0_path))

        self.wavs: List[Optional[np.ndarray]] = [None] * self.size
        self.mels: List[Optional[torch.Tensor]] = [None] * self.size
        self.lf0s: List[Optional[np.ndarray]] = [None] * self.size

    def preprocess(self, idx):
        file = self.filenames[idx]

        wav = preprocess_wav(file)
        mel = mel_spectrogram(
            torch.tensor(wav).unsqueeze(0),
            n_fft=1024,
            num_mels=80,
            sampling_rate=16000,
            hop_size=160,
            win_size=1024,
            fmin=80,
            fmax=8000,
        )[0].T
        lf0 = get_converted_lf0uv(wav, self.lf0[idx][0], self.lf0[idx][1], convert=True)

        self.wavs[idx] = wav
        self.mels[idx] = mel
        self.lf0s[idx] = lf0

    def __getitem__(self, idx):
        if self.wavs[idx] is None:
            self.preprocess(idx)
        wav = self.wavs[idx]
        mel = self.mels[idx]
        embed = self.embeds[idx]
        lf0 = self.lf0s[idx]
        return wav, mel, embed, lf0

    def __len__(self):
        return self.size


def collate_fn(data):
    batch_size = len(data)
    max_in_length = max(unit[0].shape[0] for unit in data)
    max_out_length = max(unit[1].shape[0] for unit in data)
    if max_out_length % 4 != 0:
        max_out_length += 4 - max_out_length % 4  # Align to 4

    wavs = torch.empty((batch_size, max_in_length))
    in_lengths = torch.empty(batch_size, dtype=torch.int32)
    mels = torch.empty((batch_size, max_out_length, 80))
    out_lengths = torch.empty(batch_size, dtype=torch.int32)
    lf0_uvs = torch.empty((batch_size, max_in_length // 160, 2))
    speaker_embeds = torch.empty((batch_size, 256))
    stop_tokens = torch.zeros((batch_size, max_out_length))
    for i, unit in enumerate(data):
        wav, mel, embed, lf0 = unit
        in_length = wav.shape[0]
        out_length = mel.shape[0]

        wavs[i, :in_length] = torch.from_numpy(wav)
        in_lengths[i] = in_length
        mels[i, :out_length, :] = mel[:out_length, :]
        out_lengths[i] = out_length
        lf0_uvs[i, :out_length, :] = torch.from_numpy(lf0[:out_length, :])
        speaker_embeds[i, :] = torch.from_numpy(embed)

    return wavs, in_lengths, mels, out_lengths, lf0_uvs, speaker_embeds, stop_tokens


def train(model: VCFinetune, dataset: AiTangDataset, log_root_path: Union[str, Path]):
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    Path(log_root_path).mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        default_root_dir=str(log_root_path),
    )
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    extractor, convertor = load_models(
        extractor_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\extractor\24epoch.pt",
        convertor_config_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\convertor\ppg2mel.yaml",
        convertor_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\convertor\ppg2melbest_loss_step_322000.pth",
    )
    dataset = AiTangDataset(
        root_dir=r"E:\PKU\Lecture_AI\aidatatang_200zh\aidatatang_200zh\corpus\dev",
        embeds_path=r"E:\PKU\Lecture_AI\TTL-VC\preprocessed\embeds.npy",
        lf0_path=r"E:\PKU\Lecture_AI\TTL-VC\preprocessed\lf0_mean_stds.npy",
    )
    model = VCFinetune(extractor, convertor)
    train(model, dataset, log_root_path=r"./checkpoints/vc_finetune")
