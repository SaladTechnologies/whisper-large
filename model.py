from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datetime import datetime
import torch
import os
import torchaudio

model = os.environ.get("MODEL")
model_dir = os.environ.get("MODEL_DIR")
model_path = os.path.join(model_dir, model)

torchaudio.set_audio_backend("soundfile")


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

resamplers = {}


def transcribe(inputAudio):
    t1 = datetime.now()

    tensor, samplerate = torchaudio.load(inputAudio, normalize=True)

    # if tensor.shape[0] > 1:
    # convert stereo to mono
    # tensor = torch.mean(tensor, dim=0, keepdim=True)

    if samplerate != 16000:
        if samplerate not in resamplers:
            resamplers[samplerate] = torchaudio.transforms.Resample(samplerate, 16000)
        tensor = resamplers[samplerate](tensor).squeeze()

    inputs = processor(
        tensor, return_tensors="pt", sampling_rate=16000, do_normalize=True
    )
    input_features = inputs.input_features.to("cuda").half()
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    inference_time = datetime.now() - t1
    # print(
    #     f"⌚ Response time {format_timedelta(inference_time)} for {len(tensor[0]) / 16000} seconds of audio"
    # )
    return transcription, inference_time


t1 = datetime.now()

with open("Recording.wav", "rb") as f:
    transcription, inference_time = transcribe(f)


print(
    f"⌚ Test Audio Transcribed in {format_timedelta(inference_time)}: {transcription}"
)
