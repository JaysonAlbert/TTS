ARG BASE=ghcr.io/coqui-ai/tts-cpu:latest

FROM ${BASE}
COPY ./tts_models--zh-CN--baker--tacotron2-DDC-GST /root/.local/share/tts/tts_models--zh-CN--baker--tacotron2-DDC-GST

ENTRYPOINT ["tts-server"]
CMD ["--model_name","tts_models/zh-CN/baker/tacotron2-DDC-GST"]