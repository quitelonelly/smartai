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
                
                # Запрос на транскрипцию с указанием языка
                transcript_url = "https://api.assemblyai.com/v2/transcript"
                transcript_request = {
                    "audio_url": audio_url,
                    "language_code": "ru"  # Указываем русский язык
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
                            return status_data['text']
                        elif status_data['status'] == 'failed':
                            logger.error("Транскрипция не удалась.")
                            raise HTTPException(status_code=500, detail="Транскрипция не удалась")
        
        except Exception as e:
            logger.exception("Во время транскрипции произошла ошибка.")
            raise HTTPException(status_code=500, detail=str(e))