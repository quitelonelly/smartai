import os
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import HTTPException
import aiohttp

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

async def transcribe_audio(file_obj):
    url = "https://api.assemblyai.com/v2/upload"
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Загрузка аудиофайла.")
            # Загрузка аудиофайла
            async with session.post(url, headers={"authorization": ASSEMBLYAI_API_KEY}, data=file_obj.file) as response:
                if response.status != 200:
                    logger.error("Ошибка загрузки аудиофайла: %s", response.status)
                    raise HTTPException(status_code=response.status, detail="Ошибка загрузки аудиофайла")
                
                response_data = await response.json()
                audio_url = response_data['upload_url']
                
                # Запрос на транскрипцию с указанием языка и определения говорящих
                transcript_url = "https://api.assemblyai.com/v2/transcript"
                transcript_request = {
                    "audio_url": audio_url,
                    "language_code": "ru",  # Указываем русский язык
                    "speaker_labels": True  # Указываем, что нужно определить говорящих
                }
                
                logger.info("Запрос транскрипции.")
                async with session.post(transcript_url, headers={"authorization": ASSEMBLYAI_API_KEY}, json=transcript_request) as transcript_response:
                    transcript_data = await transcript_response.json()
                    transcript_id = transcript_data['id']
                    
                    # Ожидание завершения транскрипции
                    while True:
                        await asyncio.sleep(5)  # Пауза перед повторной проверкой
                        status_response = await session.get(f"{transcript_url}/{transcript_id}", headers={"authorization": ASSEMBLYAI_API_KEY})
                        status_data = await status_response.json()
                        
                        if status_data['status'] == 'completed':
                            logger.info("Транскрипция успешно завершена.")
                            # Форматируем результат с указанием говорящих
                            formatted_transcription = format_transcription(status_data['words'])
                            return classify_roles(formatted_transcription)
                        elif status_data['status'] == 'failed':
                            logger.error("Транскрипция не удалась.")
                            raise HTTPException(status_code=500, detail="Транскрипция не удалась")
        
        except Exception as e:
            logger.exception("Во время транскрипции произошла ошибка.")
            raise HTTPException(status_code=500, detail=str(e))

def format_transcription(words):
    transcription = []
    current_sentence = []
    current_speaker = None

    for i, word in enumerate(words):
        # Если это начало или сменился спикер, начинаем новое предложение
        if current_speaker is None or word['speaker'] != current_speaker:
            if current_sentence:
                transcription.append({
                    "text": " ".join(current_sentence),
                    "speaker": current_speaker
                })
                current_sentence = []
            current_speaker = word['speaker']

        # Добавляем слово в текущее предложение
        current_sentence.append(word['text'])

        # Проверяем, заканчивается ли слово на завершающий знак препинания
        if word['text'].endswith(('.', '?', '!', '...')):
            # Если предложение завершено, добавляем его в транскрипцию
            transcription.append({
                "text": " ".join(current_sentence),
                "speaker": current_speaker
            })
            current_sentence = []  # Сбрасываем текущее предложение

    # Если остались слова в текущем предложении, добавляем их
    if current_sentence:
        transcription.append({
            "text": " ".join(current_sentence),
            "speaker": current_speaker
        })

    return transcription


def classify_roles(transcription):
    classified = []
    first_speaker = None  # Для хранения первого спикера

    # Определяем ключевые слова
    manager_keywords = ["Меня зовут", "интересовались", "давайте", "предлагаю"]
    client_keywords = ["я хочу", "мне нужно", "можете", "как"]

    for entry in transcription:
        text = entry['text']
        speaker = entry['speaker']

        # Если это первая реплика, определяем роли
        if first_speaker is None:
            first_speaker = speaker
            role = "Клиент"  # Предполагаем, что первый спикер — это клиент
        else:
            # Проверяем наличие ключевых слов
            if any(keyword in text.lower() for keyword in manager_keywords):
                role = "Менеджер"
            elif any(keyword in text.lower() for keyword in client_keywords):
                role = "Клиент"
            else:
                # Если спикер совпадает с первым, это клиент, иначе — менеджер
                role = "Клиент" if speaker == first_speaker else "Менеджер"

        classified.append({"role": role, "text": text})

    return classified