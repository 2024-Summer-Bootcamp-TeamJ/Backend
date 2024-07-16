import os
from celery import Celery
from openai import OpenAI
from langchain.prompts import PromptTemplate
from pydub import AudioSegment
from gtts import gTTS
import io
import base64
import librosa
import soundfile as sf


broker_url = os.getenv("CELERY_BROKER_URL")
app = Celery("worker", broker=broker_url, backend="rpc://")


@app.task
def add(x, y):
    return x + y


@app.task
def gpt_answer(question):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            Imagine you are {character_name},
            a wise and experienced advisor. Given the context: "{context}",
            how would you respond to this inquiry: "{question}"?',
            1줄로 말해
            (in korean)
            """,
    )

    def generate_prompt(character_name, question, context):
        return prompt_template.format(
            character_name=character_name, question=question, context=context
        )

    character_name = "오은영"
    context = ""

    prompt = generate_prompt(character_name, question, context)

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
    )

    print(completion.choices[0].message.content)

    return completion.choices[0].message.content


@app.task
def generate_audio_from_string(string, pitch, speed, frequency, low_pass_filter_value):
    os.makedirs("samples", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    result_sound = None

    for i, letter in enumerate(string):
        if letter == " ":
            new_sound = AudioSegment.silent(duration=50)
        else:
            if not os.path.isfile(f"samples/{letter}.mp3"):
                tts = gTTS(letter, lang="ko")
                tts.save(f"samples/{letter}.mp3")

            letter_sound, sr = librosa.load(f"samples/{letter}.mp3")

            # 피치 변경
            octaves = pitch
            print(f"{letter} - octaves: {octaves}")
            pitch_shifted_sound = librosa.effects.pitch_shift(
                letter_sound, sr=sr, n_steps=octaves * 12
            )

            # 주파수 변경 (목소리 굵기 변경)
            frequency_factor = frequency
            print(f"{letter} - frequency_factor: {frequency_factor}")
            frequency_scaled_sound = librosa.effects.time_stretch(
                pitch_shifted_sound, rate=frequency_factor
            )

            # 재생 속도 변경
            speed_factor = speed
            print(f"{letter} - speed_factor: {speed_factor}")
            time_stretched_sound = librosa.effects.time_stretch(
                frequency_scaled_sound, rate=speed_factor
            )

            temp_filename = f"samples/temp_{letter}.wav"
            sf.write(temp_filename, time_stretched_sound, sr)
            new_sound = AudioSegment.from_wav(temp_filename)

            # 저역 필터 적용
            new_sound = new_sound.low_pass_filter(low_pass_filter_value)

        # result_sound 초기화
        if result_sound is None:
            result_sound = new_sound
        else:
            result_sound += new_sound

    # 메모리 버퍼에 오디오 데이터 저장
    buffer = io.BytesIO()
    result_sound.export(buffer, format="mp3")
    buffer.seek(0)

    # Base64로 인코딩
    audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return audio_base64
