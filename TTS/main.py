from TTS.api import TTS
import os
os.environ['HTTP_PROXY'] = 'http://localhost:12334'
os.environ['HTTPS_PROXY'] = 'http://localhost:12334'

api = TTS("tts_models/hak/fairseq/vits")
api.tts_with_vc_to_file(
    "今天天气非常好。",
    speaker_wav="speaker.wav",
    file_path="output.wav"
)