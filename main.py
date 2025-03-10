import logging
from fastapi import FastAPI, Response, UploadFile, File
from repository import transcribe_audio

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart"
)

@app.get("/")
async def read_root():
    return {"message": "Добро пожаловать в Smart API!"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204) 

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    logger.info("Получен файл для транскрипции.")
    try:
        transcription = await transcribe_audio(file)
        logger.info("Транскрипция завершена.")
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Ошибка при транскрипции: {e}")
        return {"error": str(e)}