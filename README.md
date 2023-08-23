# whisper-large
An inference server running the Whisper Large model


## Build the container

First, you will need to download the whisper large model from the [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) model page. You can do this by running the following command:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/whisper-large-v2 ./model

# remove pointers to the weights
rm ./model/flax_model.msgpack
rm ./model/pytorch_model.bin
rm ./model/tf_model.h5
```

Now, download the actual pytorch weights with wget:

```bash
wget https://huggingface.co/openai/whisper-large-v2/resolve/main/pytorch_model.bin -P ./model
```

Now, you can build the container:

```bash
docker build -t saladtechnologies/whisper-large:latest .
```

## Run the container

To run the container, make sure you mount your GPU, and expose port 8888:

```bash
docker run \
--gpus all \
-p 1111:1111 \
-e HOST="0.0.0.0" \
saladtechnologies/whisper-large:latest
```

## Use The Container

```bash
curl  -X POST \
  'http://localhost:1111/generate/' \
  --header 'Content-Type: application/octet-stream' \
  --data-binary '@/home/shawn/code/SaladTechnologies/whisper-large/Recording.wav'
```