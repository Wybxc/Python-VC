from tts import *
from utils.audio import save

load_models(
    synthesizer_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\tacotron\pretrained-11-7-21_75k.pt",
    embed_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\embed\pretrained.pt",
    hifigan_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\hifigan\g_hifigan.pt",
)

s = "两个黄鹂鸣翠柳，一行白鹭上青天"
embed = speaker_embed(r"E:\Python\MockingBird\data_set\ke_qing.mp3")
spec, br = synthesize(s, embed)
wav, sr = vocode(spec, br)
save(f"./{s}", wav, sr)
