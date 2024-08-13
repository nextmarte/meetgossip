import streamlit as st
from pydub import AudioSegment
import speech_recognition as sr
from concurrent.futures import ProcessPoolExecutor
import time


def transcrever_chunk(chunk):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(
        chunk.raw_data, chunk.frame_rate, chunk.sample_width)
    try:
        texto = recognizer.recognize_google(audio_data, language="pt-BR")
        return texto
    except sr.UnknownValueError:
        return "[Inaudível]"
    except sr.RequestError as e:
        return "[Erro de solicitação]"


def process_audio(file):
    audio = AudioSegment.from_file(file, format="m4a")
    chunk_length_ms = 60000  # 1 minuto
    chunks = [audio[i:i + chunk_length_ms]
              for i in range(0, len(audio), chunk_length_ms)]

    transcriptions = []
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    with ProcessPoolExecutor() as executor:
        results = executor.map(transcrever_chunk, chunks)
        for i, result in enumerate(results):
            transcriptions.append(result)
            progress_bar.progress((i + 1) / total_chunks)
            time.sleep(0.1)  # Simulação de tempo de processamento

    return transcriptions


st.title("MeetGossip - transcritor de áudio .m4a para texto")
st.write("Faça upload de um arquivo de áudio em formato .m4a para transcrição.")

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type="m4a")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/m4a')
    if st.button("Iniciar Transcrição"):
        with st.spinner('Transcrevendo...'):
            transcriptions = process_audio(uploaded_file)
            st.success('Transcrição concluída!')
            transcricao_texto = " ".join(transcriptions)
            st.write(transcricao_texto)
            with open("transcricao.txt", "w") as f:
                f.write(transcricao_texto)
            st.download_button('Baixar Transcrição', data=transcricao_texto,
                               file_name='transcricao.txt', mime='text/plain')

st.write("Desenvolvido por Nextmarte: marcusantonio@id.uff.br")
