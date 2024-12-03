import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment

# 기존 함수 및 클래스들은 그대로 유지
def scatter_plot(audio_features, probabilities):
    """
    Generates a scatter plot to visualize the relationship between audio features and probabilities.

    Args:
        audio_features (dict): Dictionary containing audio features.
        probabilities (list): List of deepfake probabilities for each audio file.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot scatter points
    for feature, values in audio_features.items():
        ax.scatter(values, probabilities, label=feature, alpha=0.7)
    
    # Add labels and legend
    ax.set_title("Audio Features vs Deepfake Probabilities")
    ax.set_xlabel("Audio Features")
    ax.set_ylabel("Deepfake Probability")
    ax.legend()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

def setStreamlitGUI():
    st.set_page_config(layout="wide")
    st.title("리얼보이스 - AI 딥보이스 식별 솔루션")
    st.info("AI 음성 식별 모델을 활용하여 음성의 진위 여부를 판별합니다.")
    st.warning("가급적 소음이 적고 한 명의 화자만 포함된 음성 파일을 업로드 해주시길 바랍니다.")

    col1, col2 = st.columns(2)

    # 로컬 mp3 파일 업로드 후 모델에 입력
    with col1:
        st.info("로컬 컴퓨터에서 .mp3 파일을 업로드하여 분석하세요.")
        uploaded_file = st.file_uploader("파일 선택", type='mp3')

        if uploaded_file is not None:
            if st.button("오디오 분석"):
                st.info("오디오 분석 중...")

                # 예제 데이터를 생성
                audio_features = {
                    "Sampling Rate (Hz)": [22000, 44100, 16000, 32000, 22050],
                    "Noise Level (%)": [10, 5, 15, 8, 12],
                    "Duration (s)": [5, 3, 7, 4, 6]
                }
                probabilities = [0.2, 0.8, 0.5, 0.1, 0.3]

                st.info("분석 결과 및 시각화")
                scatter_plot(audio_features, probabilities)

    # 사용자 라이브 오디오 녹음 후 모델에 입력
    with col2:
        st.info("5초 동안 라이브 오디오를 녹음하여 분석하세요.")
        if st.button("오디오 녹음"):
            st.info("녹음된 오디오 데이터를 처리 중...")
            # 녹음 및 처리 로직 추가 가능
            # pyaudioStream()

def main():
    setStreamlitGUI()

if __name__ == "__main__":
    main()
