import os
import asyncio
import logging
import json
from dotenv import load_dotenv
from fastapi import HTTPException
import aiohttp
import httpx

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Загружаем API-ключи
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Модель DeepSeek R1
DEEPSEEK_MODEL = "deepseek/deepseek-r1"

# Загружаем ключевые фразы из JSON-файла
with open("key_phrases.json", "r", encoding="utf-8") as file:
    key_phrases = json.load(file)

MANAGER_PHRASES = key_phrases["manager_phrases"]
CLIENT_PHRASES = key_phrases["client_phrases"]

# Счетчик реплик для определения порядка приветствий
greeting_counter = 0

# Функция для запроса к DeepSeek R1 через OpenRouter
async def deepseek_request(text, context=None):
    """
    Отправляет запрос к модели DeepSeek для определения роли (менеджер/клиент).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Формируем контекст для модели
    messages = []
    if context:
        for entry in context:
            messages.append({
                "role": "system",
                "content": f"{entry['role']}: {entry['text']}"
            })
    
    # Добавляем текущую фразу для классификации
    messages.append({
        "role": "user",
        "content": f"""
        Ты опытный ИИ агент, который анализирует диалоги между менеджером и клиентом. Твоя задача — определить, кто говорит в следующей фразе: менеджер или клиент.

        Контекст диалога:
        {context if context else "Контекст отсутствует."}

        Учти, что первое приветствие всегда от менеджера.

        Теперь определи роль для следующей фразы:
        Текст: "{text}"
        Роль:

        Верни только одно слово: "Менеджер" или "Клиент". Ничего больше не пиши. Не добавляй рассуждения.
        """
    })
    
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка HTTP при запросе к DeepSeek: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Ошибка при запросе к DeepSeek: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def predict_role_with_deepseek(text, context=None):
    """
    Определяет роль (менеджер/клиент) для текста с использованием DeepSeek и контекста.
    """
    global greeting_counter

    try:
        # Проверяем ключевые фразы менеджера
        if any(phrase in text.lower() for phrase in MANAGER_PHRASES):
            return "Менеджер"

        # Проверяем ключевые фразы клиента
        if any(phrase in text.lower() for phrase in CLIENT_PHRASES):
            return "Клиент"

        # Если ключевые фразы не найдены, используем порядок приветствий
        if "здравствуйте" in text.lower() or "добрый день" in text.lower():
            greeting_counter += 1
            if greeting_counter == 1:
                return "Менеджер"  # Первое приветствие — менеджер
            elif greeting_counter == 2:
                return "Клиент"  # Второе приветствие — клиент

        # Если порядок приветствий не помогает, используем DeepSeek с контекстом
        response = await deepseek_request(text, context)
        logger.info(f"Ответ DeepSeek: {response}")  # Логируем полный ответ
        
        # Извлекаем роль из поля content
        role = response["choices"][0]["message"]["content"].strip()
        
        # Очищаем ответ: оставляем только "Менеджер" или "Клиент"
        if "Менеджер" in role:
            return "Менеджер"
        elif "Клиент" in role:
            return "Клиент"
        else:
            logger.warning(f"DeepSeek вернул неожиданную роль: '{role}'. Использую значение по умолчанию.")
            return "Клиент"  # По умолчанию
    except Exception as e:
        logger.error(f"Ошибка при определении роли: {e}")
        return "Клиент"  # По умолчанию

async def classify_roles_with_deepseek(transcription):
    """
    Классифицирует роли для каждой фразы в транскрипции с использованием DeepSeek.
    """
    classified = []
    context = []  # Сохраняем контекст диалога

    for entry in transcription:
        text = entry['text']
        
        # Определяем роль с учетом всего контекста
        role = await predict_role_with_deepseek(text, context)
        logger.info(f"Текст: '{text}' -> Роль: '{role}'")  # Логируем результат
        
        # Добавляем текущую реплику в контекст
        context.append({"role": role, "text": text})
        
        # Сохраняем результат
        classified.append({"role": role, "text": text})

    return classified

async def transcribe_audio(file_obj):
    """
    Транскрибирует аудиофайл с помощью AssemblyAI и возвращает транскрипцию.
    """
    upload_url = "https://api.assemblyai.com/v2/upload"
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.info("Загрузка аудиофайла.")
            async with session.post(
                upload_url,
                headers={"authorization": ASSEMBLYAI_API_KEY},
                data=file_obj.file
            ) as upload_response:
                if upload_response.status != 200:
                    logger.error("Ошибка загрузки аудиофайла: %s", upload_response.status)
                    raise HTTPException(status_code=upload_response.status, detail="Ошибка загрузки аудиофайла")
                
                upload_data = await upload_response.json()
                audio_url = upload_data['upload_url']
                
                logger.info("Запрос транскрипции.")
                transcript_request = {
                    "audio_url": audio_url,
                    "language_code": "ru",
                    "speaker_labels": True
                }
                
                async with session.post(
                    transcript_url,
                    headers={"authorization": ASSEMBLYAI_API_KEY},
                    json=transcript_request
                ) as transcript_response:
                    transcript_data = await transcript_response.json()
                    transcript_id = transcript_data['id']
                    
                    while True:
                        await asyncio.sleep(5)
                        status_response = await session.get(
                            f"{transcript_url}/{transcript_id}",
                            headers={"authorization": ASSEMBLYAI_API_KEY}
                        )
                        status_data = await status_response.json()
                        
                        if status_data['status'] == 'completed':
                            logger.info("Транскрипция успешно завершена.")
                            formatted_transcription = format_transcription(status_data['words'])
                            roles = await classify_roles_with_deepseek(formatted_transcription)
                            return roles
                        elif status_data['status'] == 'failed':
                            logger.error("Транскрипция не удалась.")
                            raise HTTPException(status_code=500, detail="Транскрипция не удалась")
        
        except Exception as e:
            logger.exception("Во время транскрипции произошла ошибка.")
            raise HTTPException(status_code=500, detail=str(e))

def format_transcription(words):
    """
    Форматирует транскрипцию в список реплик.
    """
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