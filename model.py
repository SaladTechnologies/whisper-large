from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import datetime
import torch
import os
import torchaudio
from typing import BinaryIO

model = os.environ.get("MODEL")
model_dir = os.environ.get("MODEL_DIR")
model_path = os.path.join(model_dir, model)


def format_timedelta(td):
    seconds = td.total_seconds()
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if seconds < 1:
        return "<1 sec"
    return "{} {} {} {}".format(
        "" if int(days) == 0 else str(int(days)) + " days",
        "" if int(hours) == 0 else str(int(hours)) + " hours",
        "" if int(minutes) == 0 else str(int(minutes)) + " mins",
        "" if int(seconds) == 0 else str(int(seconds)) + " secs",
    )


t1 = datetime.now()
processor = AutoProcessor.from_pretrained(
    model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
)
print("⌚ Model Processor created", format_timedelta(datetime.now() - t1))

t1 = datetime.now()
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
)
print("⌚ Model loaded (.from_pretrained)", format_timedelta(datetime.now() - t1))

t1 = datetime.now()

model.cuda()

print("⌚ Model .cuda()", format_timedelta(datetime.now() - t1))

t1 = datetime.now()


def eval(inputAudio: BinaryIO):
    t1 = datetime.now()

    inputAudio = torchaudio.load(inputAudio)[0]
    inputs = processor(inputAudio, return_tensors="pt")
    input_features = inputs.input_features
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    inference_time = datetime.now() - t1
    print(
        f"⌚ Response time {format_timedelta(inference_time)} in len: { len(input.text) } resp len { len(resp) }"
    )
    return transcription, inference_time
