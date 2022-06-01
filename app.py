import os

from typing import BinaryIO
import streamlit as st
import tempfile

from utils.audio import save
from vc import *

st.set_page_config(  # type: ignore
    layout="wide",
    menu_items={
        "About": "人工智能引论课程作业。",
    },
)

# 加载声音，返回临时文件名字
def create_sound_tmp(file: BinaryIO):
    # 创建临时文件
    temp = tempfile.NamedTemporaryFile(mode="w+b", suffix=".wav", delete=False)
    temp.close()
    try:
        # 二进制打开临时写文件，并将file中的数据写到临时文件中
        with open(temp.name, "wb") as f:
            f.write(file.read())
            f.close()
        return temp.name
    finally:
        pass


# 返回声音中的数据，用于st.audio读取数据播放音频
def export_sound(str) -> bytes:
    with open(str, "rb") as st:
        return st.read()


def 加载模型():
    if not is_model_loaded():
        load_models(
            extractor_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\extractor\24epoch.pt",
            convertor_config_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\convertor\ppg2mel.yaml",
            convertor_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\convertor\ppg2melbest_loss_step_322000.pth",
            embed_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\embed\pretrained.pt",
            hifigan_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\hifigan\g_hifigan_24k.pt",
            hifigan_config_path=r"E:\PKU\Lecture_AI\TTL-VC\pre_trained\hifigan\config_24k.json",
        )


# 开始绘制
st.header("语音转换")

# 区分为两个独立的块
left, right = st.columns(2)

source_path = target_path = None


with left:
    if file_src := st.file_uploader("选择源音频", type=["mp3", "wav", "flac", "ogg", "aac"]):
        source_path = create_sound_tmp(file_src)
        with st.expander("音频详情"):
            st.audio(export_sound(source_path), format="audio/wav")
    else:
        source_path = None

    if file_dst := st.file_uploader(
        "选择目标音频", type=["mp3", "wav", "flac", "ogg", "aac"]
    ):
        target_path = create_sound_tmp(file_dst)
        with st.expander("音频详情"):
            st.audio(export_sound(target_path), format="audio/wav")
    else:
        target_path = None

# 转换+结果展示
if source_path and target_path:
    with right:
        if st.button("开始转换"):
            with st.spinner("正在转换..."):
                加载模型()
                # 转换过程
                source = source_path
                target = VoiceTarget(target_path)
                spec, br = convert(source, target)
                wav, sr = vocode(spec, br)

                # 创建临时文件储存结果
                temp = tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".wav", delete=False
                )
                temp.close()
                try:
                    save(temp.name, wav, sr)
                    del wav # 释放内存
                    
                    st.success("转换成功!")
                    st.write("变声结果：")
                    byte = export_sound(temp.name)
                    st.audio(byte, format="audio/wav")
                    st.download_button(
                        label="下载变声结果，保存为.wav格式",
                        data=byte,
                        file_name="result.wav",
                        mime="audio/wav",
                    )
                finally:
                    os.remove(temp.name)


# 收尾清理工作
if source_path:
    os.remove(source_path)
if target_path:
    os.remove(target_path)

with st.spinner("加载模型中……"):
    加载模型()
