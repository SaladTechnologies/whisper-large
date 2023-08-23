from fastapi import FastAPI, Request
from io import BytesIO
import model

app = FastAPI()


@app.post("/generate/")
async def generate(request: Request):
    file: bytes = await request.body()
    try:
        output, inference_time = model.transcribe(BytesIO(file))
        return {"text": output, "inference_time": str(inference_time)}
    except Exception as e:
        return {"error": str(e)}


@app.get("/hc")
async def health_check():
    return "ok"
