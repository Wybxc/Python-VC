import matplotlib
import matplotlib.pyplot as plt

from utils.audio import save
from vc import *

matplotlib.use("TkAgg")

load_models(
    extractor_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\extractor\24epoch.pt",
    convertor_config_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\convertor\ppg2mel.yaml",
    convertor_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\convertor\ppg2melbest_loss_step_322000.pth",
    embed_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\embed\pretrained.pt",
    hifigan_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\hifigan\g_hifigan_24k.pt",
    hifigan_config_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\hifigan\config_24k.json",
)


source = r"E:\Python\MockingBird\data_set\xiao_gong.mp3"
# source = r"E:\QQ\1779724477\FileRecv\MobileFile\5月23日 下午9点26分.aac"
# target = VoiceTarget(r"E:\Python\MockingBird\data_set\qiqi.mp3")
target = VoiceTarget(r"E:\OneDrive\文档\录音\录音 (2).m4a")
spec, br = convert(source, target)
# plt.imshow(spec, aspect="auto", origin="lower", cmap="inferno")
# plt.show()
wav, sr = vocode(spec, br)
save("./vc_test", wav, sr)
