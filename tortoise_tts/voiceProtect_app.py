# '##::: ##::'#######::'########:'########:
#  ###:: ##:'##.... ##:... ##..:: ##.....::
#  ####: ##: ##:::: ##:::: ##:::: ##:::::::
#  ## ## ##: ##:::: ##:::: ##:::: ######:::
#  ##. ####: ##:::: ##:::: ##:::: ##...::::
#  ##:. ###: ##:::: ##:::: ##:::: ##:::::::
#  ##::. ##:. #######::::: ##:::: ########:
# ..::::..:::.......::::::..:::::........::

# Both local deployment and the live Streamlit deployment launch from THIS FILE (./tortoise_tts/voiceProtect_app.py), NOT (./voiceProtect_app.py).
# Deploying from the root directory causes threading issues as tortoise_tts must be launched in the main thread, which is assigned to the parent dir of __file__ on launch.
# ER: ValueError: signal only works in main thread of the main interpreter ; how to fix, or otherwise how to avoid having to download locally. See Issues.txt for full trace.

# //////////////////////////////////////////////////////////////////////////////

# PHASED OUT: Try to move script into tortoise_tts module while executing - threading issues:
# absolute_path = os.path.abspath(__file__)
# subdir_path = os.path.abspath("./tortoise_tts")
# target_path = os.path.join(subdir_path, "voiceProtect_app.py")

# st.info(absolute_path)
# st.info(target_path)

# os.rename(absolute_path, target_path)
# os.replace(absolute_path, target_path)
# shutil.move(absolute_path, target_path)

# //////////////////////////////////////////////////////////////////////////////

# System imports
import wave
import sys
import os 
import io 
from PIL import Image
from glob import glob
import subprocess

# Add tortoise_tts directory to path:
# Calculate the absolute path of the directory containing app.py, submodule directory, and append to system path
app_directory = os.path.dirname(os.path.abspath(__file__))
submodule_directory = os.path.join(app_directory, 'tortoise_tts')
sys.path.append(submodule_directory)

# Add root directory to sys path
# Used for non-package imports (inputAudio.py, configWavPlot.py)
parent_directory = os.path.join(os.path.dirname(__file__), '..')
parent_directory = os.path.abspath(parent_directory)
sys.path.append(parent_directory)

# Set current working directory to parent dir of main app file(/tortoise_tts)
# Resolves issue: local deploy launches CWD in parent dir, Streamlit Deploy launches CWD in root dir
os.chdir(app_directory)

# Main functionality
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import pyaudio 

import torchaudio
import torch
import torch.nn.functional as F
import streamlit as st

# Misc imports
import librosa
import plotly.express as px
import numpy as np 
from scipy.io.wavfile import read 
from pydub import AudioSegment
import matplotlib.pyplot as plt

# App function imports
from inputAudio import inputAudio
from configWavPlot import configWavPlot

# Removed backend matplotlib GUI - caused multithreading issues
# import matplotlib
# matplotlib.use("TkAgg")  # Change to "Qt5Agg" or "QtAgg" if you prefer

# load an audio file as tensor for model input
def load_audio(audiopath, sampling_rate =22000):
    if isinstance(audiopath, str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Error for unsupported audio format {audiopath[-4]}"
    elif isinstance(audiopath, io.BytesIO):

        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0]
    
    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    
    #torch.any(param) returns true if any param is true
    if torch.any(audio > 2) or not torch.any(audio < 0 ):
        print(f"Error with audio data, Max: {audio.max()} and min: {audio.min()}")
        audio.clip_(-1, 1)

    return audio.unsqueeze(0)

# function to classify audio with tortoise-tts library
def classify_audio_clip(clip):

    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim = 1, embedding_dim = 512, depth = 5, downsample_factor = 4,
                                                    resnet_blocks = 2, attn_blocks = 4, num_attn_heads = 4, base_channels = 32,
                                                    dropout = 0, kernel_size = 5, distribute_zero_label= False)
    state_dict = torch.load('../classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze(dim = 0)
    results = F.softmax(classifier(clip), dim= -1)
    return results[0][0]

# Create Streamlit app layout
def setStreamlitGUI():

    # 앱 GUI
    # st.info("현재 작업 디렉토리:", os.getcwd())

    st.set_page_config(layout="wide")

    logo = Image.open('../resources/VoiceProtect-logo.png')
    # 로고 크기 조정, 비율 유지
    logo.thumbnail((600, 600))

    # st.image(logo, width=50, use_column_width="auto")

    st.title("리얼보이스 - AI 딥보이스 식별 솔루션")
    st.info("AI 음성 식별 모델을 활용하여 음성의 진위 여부를 판별합니다.")
    st.warning("가급적 소음이 적고 한 명의 화자만 포함된 음성 파일을 업로드 해주시길 바랍니다. 정확한 결과를 보장하지 않을 수 있습니다.")
    # st.info("https://huggingface.co/jbetker/tortoise-tts-v2")

    col1, col2 = st.columns(2)

    # 로컬 mp3 파일 업로드 후 모델에 입력
    with col1:
        st.info("로컬 컴퓨터에서 .mp3 파일을 업로드하여 분석하세요.")
        uploaded_file = st.file_uploader("파일 선택", type='mp3')

        # Streamlit 배포 문제 해결
        # st.info(uploaded_file)
        # st.info(type(uploaded_file))

        if uploaded_file is not None:

            if st.button("오디오 분석"):

                st.info("오디오 분석 중...")
                row1, row2, row3 = st.columns(3)

                with row1:
                    audio_clip = load_audio(uploaded_file)
                    results = classify_audio_clip(audio_clip)
                    results = results.item()

                    st.info("분석 결과:")
                    st.info(f"딥페이크 오디오일 확률:  {results}")
                    st.success(f"업로드된 오디오는 {results * 100: .2f}% 확률로 AI가 생성한 것으로 판단됩니다.")

                with row2:
                    st.info("오디오 플레이어")
                    st.audio(uploaded_file)

                with row3:
                    # 업로드된 mp3를 wav로 변환, matplotlib용 wav 경로
                    output_wav_file = '../resources/upload.wav'
                    AudioSegment.from_mp3(uploaded_file).export(output_wav_file, format="wav")
                    absolute_path = os.path.abspath(output_wav_file)

                    generateWavePlot(absolute_path)

    # 사용자 라이브 오디오 녹음 후 모델에 입력
    with col2:

        # ST 배포용 장치 테스트
        # p = pyaudio.PyAudio()
        # num = p.get_device_count()
        # st.info(num)
        # for i in range(p.get_device_count()):
        #     info = p.get_device_info_by_index(i)
        #     st.info(f"장치 {i}: {info['name']}")

        st.info("5초 동안 라이브 오디오를 녹음하여 분석하세요.")

        if st.button("오디오 녹음"):

            # 오디오를 녹음하고 "output.wav" 파일로 저장
            pyaudioStream()

            row1, row2, row3 = st.columns(3)

            with row1:
                # MP3 경로 (matplotlib용)
                relative_path = "../resources/output.mp3"
                absolute_path = os.path.abspath(relative_path)

                audio_clip = load_audio(absolute_path)
                results = classify_audio_clip(audio_clip)
                results = results.item()

                st.info("분석 결과:")
                st.info(f"딥페이크 오디오일 확률: {results}")
                st.info(f"녹음된 오디오는 {results * 100: .2f}% 확률로 AI가 생성한 것으로 판단됩니다.")

            with row2:
                st.info("오디오 플레이어")
                st.audio(absolute_path)

            with row3:
                # WAV 경로 (matplotlib용)
                relative_path = "../resources/output.wav"
                absolute_path = os.path.abspath(relative_path)

                generateWavePlot(absolute_path)



# Record user audio stream with pyaudio
# Convert to mp3 using pydub (load_audio() accepts mp3)
def pyaudioStream():
    inputAudio.pyaudioStream()


# Generate and stylize waveform plot from input .WAV path
def generateWavePlot(path: str):
    configWavPlot.generateWavePlot(path)


# Run script
def main():
    setStreamlitGUI()

if __name__ == "__main__":
    main()

