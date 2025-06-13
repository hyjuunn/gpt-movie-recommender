from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates # pip install jinja2
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import asyncio
import json
from io import BytesIO
import wave
from dotenv import load_dotenv
import os

app = FastAPI()


# Set up template directory
templates = Jinja2Templates(directory="templates")

# Set up static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load API key
load_dotenv() # Load .env file
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key = api_key
)

# Function to read prompt content from file
def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        prompt = file.read().strip()
    return prompt

movie_prompt = load_prompt("movie_prompt.txt")


messages = [{"role": "system","content": movie_prompt}]


# Define request body model
class ChatRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat_stream")
async def chat_stream(prompt: str = Query(...)):
    # Reset conversation to initial state with system prompt
    current_messages = [{"role": "system", "content": movie_prompt}]
    # Add new user message
    current_messages.append({"role": "user", "content": prompt})

    async def event_stream():
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=current_messages,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                data = f"data: {json.dumps({'status': 'processing', 'data': content}, ensure_ascii=False)}\n\n"
                yield data
                await asyncio.sleep(0)  # Allow other tasks to be performed (context switching)

        yield f"data: {json.dumps({'status': 'complete', 'data': 'finished'}, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }

    return StreamingResponse(event_stream(), headers=headers)



# Function to convert PCM data to WAV
def pcm_to_wav(pcm_data):
    wav_io = BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(24000)  # 24 kHz sample rate
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
    return wav_io


# Audio streaming endpoint
@app.get("/audio_stream")
async def audio_stream(prompt: str = Query(..., description="Text to convert to speech")):
    # Reset conversation to initial state with system prompt
    current_messages = [{"role": "system", "content": movie_prompt}]
    # Add new user message
    current_messages.append({"role": "user", "content": prompt})

    chat_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=current_messages,
    )
    text = chat_response.choices[0].message.content

    async def audio_event_stream():
        audio_data = BytesIO()
        with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text,
                response_format="wav"
        ) as response:
            for chunk in response.iter_bytes(1024):
                audio_data.write(chunk)
                await asyncio.sleep(0)

        audio_data.seek(0)
        return audio_data

    wav_stream = await audio_event_stream()

    return StreamingResponse(wav_stream, media_type="audio/wav")


# Text-to-speech conversion endpoint
@app.post("/convert_text_to_speech")
async def convert_text_to_speech(request: Request):
    data = await request.json()
    text = data.get("text")

    # TTS conversion code
    audio_data = BytesIO()
    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="wav"
    ) as response:
        for chunk in response.iter_bytes(1024):
            audio_data.write(chunk)
            await asyncio.sleep(0)

    audio_data.seek(0)
    return StreamingResponse(audio_data, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
