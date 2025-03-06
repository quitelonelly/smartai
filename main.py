from fastapi import FastAPI, UploadFile, File
from repository import transcribe_audio

app = FastAPI(
    title="Smart"
)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    transcription = await transcribe_audio(file)
    return {"transcription": transcription}