import os
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import HTTPException
import aiohttp
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Загрузка ключевых фраз из JSON-файла
def load_key_phrases():
    try:
        with open("key_phrases.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("Файл key_phrases.json не найден.")
        raise HTTPException(status_code=500, detail="Файл key_phrases.json не найден.")
    except json.JSONDecodeError:
        logger.error("Ошибка декодирования JSON в файле key_phrases.json.")
        raise HTTPException(status_code=500, detail="Ошибка декодирования JSON в файле key_phrases.json.")
    except Exception as e:
        logger.error(f"Ошибка загрузки ключевых фраз: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки ключевых фраз: {e}")

# Загружаем ключевые фразы
try:
    key_phrases = load_key_phrases()
except HTTPException:
    # В случае ошибки загрузки фраз, используем пустые списки
    key_phrases = {
        "manager_phrases": [],
        "client_phrases": []
    }

def predict_role(text, previous_role=None, is_first_message=False):
    """
    Определяет роль (Менеджер или Клиент) на основе текста, предыдущей роли и контекста.
    """
    # Приводим текст к нижнему регистру для сравнения
    text_lower = text.lower()

    # Если это первая реплика и текст содержит приветствие, это менеджер
    if is_first_message and any(phrase in text_lower for phrase in ["здравствуйте", "добрый день", "добрый вечер"]):
        return "Менеджер"

    # Проверяем, содержит ли текст фразы менеджера
    for phrase in key_phrases["manager_phrases"]:
        if phrase in text_lower:
            return "Менеджер"

    # Проверяем, содержит ли текст фразы клиента
    for phrase in key_phrases["client_phrases"]:
        if phrase in text_lower:
            return "Клиент"

    # Если ни одна фраза не подошла, чередуем роли
    if previous_role == "Менеджер":
        return "Клиент"
    else:
        return "Менеджер"

async def transcribe_audio(file_obj):
    url = "https://api.assemblyai.com/v2/upload"
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Загрузка аудиофайла.")
            async with session.post(url, headers={"authorization": ASSEMBLYAI_API_KEY}, data=file_obj.file) as response:
                if response.status != 200:
                    logger.error("Ошибка загрузки аудиофайла: %s", response.status)
                    raise HTTPException(status_code=response.status, detail="Ошибка загрузки аудиофайла")
                
                response_data = await response.json()
                audio_url = response_data['upload_url']
                
                transcript_url = "https://api.assemblyai.com/v2/transcript"
                transcript_request = {
                    "audio_url": audio_url,
                    "language_code": "ru",
                    "speaker_labels": True
                }
                
                logger.info("Запрос транскрипции.")
                async with session.post(transcript_url, headers={"authorization": ASSEMBLYAI_API_KEY}, json=transcript_request) as transcript_response:
                    transcript_data = await transcript_response.json()
                    transcript_id = transcript_data['id']
                    
                    while True:
                        await asyncio.sleep(5)
                        status_response = await session.get(f"{transcript_url}/{transcript_id}", headers={"authorization": ASSEMBLYAI_API_KEY})
                        status_data = await status_response.json()
                        
                        if status_data['status'] == 'completed':
                            logger.info("Транскрипция успешно завершена.")
                            formatted_transcription = format_transcription(status_data['words'])
                            roles = classify_roles(formatted_transcription)
                            return roles
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
        if current_speaker is None or word['speaker'] != current_speaker:
            if current_sentence:
                transcription.append({
                    "text": " ".join(current_sentence),
                    "speaker": current_speaker
                })
                current_sentence = []
            current_speaker = word['speaker']

        current_sentence.append(word['text'])

        if word['text'].endswith(('.', '?', '!', '...')):
            transcription.append({
                "text": " ".join(current_sentence),
                "speaker": current_speaker
            })
            current_sentence = []

    if current_sentence:
        transcription.append({
            "text": " ".join(current_sentence),
            "speaker": current_speaker
        })

    return transcription

def classify_roles(transcription):
    classified = []
    previous_role = None

    for i, entry in enumerate(transcription):
        text = entry['text']
        speaker = entry['speaker']
        
        # Определяем, является ли это первой репликой
        is_first_message = (i == 0)
        
        # Определяем роль на основе текста, предыдущей роли и контекста
        role = predict_role(text, previous_role, is_first_message)
        classified.append({"role": role, "text": text})
        
        # Обновляем предыдущую роль
        previous_role = role

    return classified