from fastapi import FastAPI, UploadFile
import model

app = FastAPI()


@app.post("/generate/")
async def generate(input: UploadFile):
    try:
        output, inference_time = model.eval(input.file)
        return {"text": output, "inference_time": str(inference_time)}
    except Exception as e:
        return {"error": str(e)}
