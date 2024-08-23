import os
from dotenv import load_dotenv
import streamlit as st
from pydub import AudioSegment
import speech_recognition as sr
from concurrent.futures import ProcessPoolExecutor
import time
import moviepy.editor as mp
import tempfile
import google.generativeai as genai
import re

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar a API do Google Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configuração do modelo
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def transcrever_chunk(chunk):
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.pause_threshold = 0.8
    recognizer.operation_timeout = None

    audio_data = sr.AudioData(chunk.raw_data, chunk.frame_rate, chunk.sample_width)
    try:
        texto = recognizer.recognize_google(audio_data, language="pt-BR")
        return texto
    except sr.UnknownValueError:
        return "[Inaudível]"
    except sr.RequestError as e:
        return "[Erro de solicitação]"

def process_audio(file, file_type):
    temp_audio_path = None
    temp_video_path = None
    try:
        if file_type == "m4a":
            audio = AudioSegment.from_file(file, format="m4a")
        elif file_type == "mp4":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
                    temp_video_file.write(file.read())
                    temp_video_path = temp_video_file.name
                video = mp.VideoFileClip(temp_video_path)
                video.audio.write_audiofile(temp_audio_file.name)
                temp_audio_path = temp_audio_file.name
            audio = AudioSegment.from_file(temp_audio_path, format="mp3")
        elif file_type == "mp3":
            audio = AudioSegment.from_file(file, format="mp3")
        else:
            raise ValueError("Formato de arquivo não suportado")

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        chunk_length_ms = 60000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        transcriptions = []
        total_chunks = len(chunks)
        progress_bar = st.progress(0)
        with ProcessPoolExecutor() as executor:
            results = executor.map(transcrever_chunk, chunks)
            for i, result in enumerate(results):
                transcriptions.append(result)
                progress_bar.progress((i + 1) / total_chunks)
                time.sleep(0.5)

        return transcriptions
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)

def clean_text(text):
    # Remove caracteres especiais e múltiplos espaços
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def summarize_meeting(transcription_text):
    try:
        # Limpar o texto de transcrição
        clean_transcription_text = clean_text(transcription_text)

        response = model.generate_content(f"Resuma este texto detalhadamente: {clean_transcription_text}")

        return response.text
    except Exception as e:
        st.error(f"Erro ao gerar sumarização: {e}")
        return None

st.title("CID@MeetGossip - áudio para texto")
st.write("Faça upload de um arquivo de áudio em formato .m4a, .mp4 ou .mp3 para transcrição.")

uploaded_file = st.file_uploader("Escolha um arquivo de áudio", type=["m4a", "mp4", "mp3"])

transcricao_texto = None  # Inicializa a variável fora do bloco if

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.audio(uploaded_file, format='audio/m4a' if file_name.endswith('.m4a') else 'audio/mp4' if file_name.endswith('.mp4') else 'audio/mp3')
    if st.button("Iniciar Transcrição"):
        with st.spinner('Transcrevendo...'):
            file_type = "m4a" if file_name.endswith('.m4a') else "mp4" if file_name.endswith('.mp4') else "mp3"
            transcriptions = process_audio(uploaded_file, file_type)
            st.success('Transcrição concluída!')
            transcricao_texto = " ".join(transcriptions)

            # Salvar a transcrição completa
            with open("transcricao.txt", "w") as f:
                f.write(transcricao_texto)

            # Permitir o download da transcrição completa
            st.download_button('Baixar Transcrição', data=transcricao_texto,
                               file_name='transcricao.txt', mime='text/plain')

            # Gerar a sumarização
            summary = summarize_meeting(transcricao_texto)
            if summary and summary.strip():
                st.subheader("Sumarização Detalhada")
                st.write(summary)
                st.download_button('Baixar Sumarização', data=summary,
                                   file_name='sumarizacao.txt', mime='text/plain')
            else:
                st.error("Falha ao gerar a sumarização. A resposta da API está vazia ou inválida.")

st.write("Desenvolvido por Nextmarte: marcusantonio@id.uff.br")