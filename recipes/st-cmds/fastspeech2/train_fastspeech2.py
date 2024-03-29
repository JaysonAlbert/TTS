import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager
from TTS.tts.utils.text.characters import IPAPhonemes, _phonemes
from TTS.tts.utils.text.phonemizers.zh_cn_phonemizer import _DEF_ZH_PUNCS

os.environ['HTTP_PROXY'] = 'http://localhost:12334'
os.environ['HTTPS_PROXY'] = 'http://localhost:12334'

output_path = os.path.dirname(os.path.abspath(__file__))

_tone = "12345。，"
characters = IPAPhonemes(characters =_tone + _phonemes + _DEF_ZH_PUNCS, punctuations=_DEF_ZH_PUNCS).to_config()

# init configs
dataset_config = BaseDatasetConfig(
    formatter="stcmds",
    path=os.path.join(output_path, "../st-cmds/"),
)

audio_config = BaseAudioConfig(
    sample_rate=16000,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = Fastspeech2Config(
    run_name="fastspeech2_stcmds",
    audio=audio_config,
    batch_size=128,
    eval_batch_size=64,
    num_loader_workers=64,
    num_eval_loader_workers=32,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    compute_energy=True,
    energy_cache_path=os.path.join(output_path, "energy_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="chinese_mandarin_cleaners",
    use_phonemes=True,
    phoneme_language="zh-cn",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    precompute_num_workers=32,
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    max_seq_len=500000,
    output_path=output_path,
    datasets=[dataset_config],
    characters = characters,
)

# compute alignments
if not config.model_args.use_aligner:
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/zh-CN/stcmds/fastspeech2")
    # TODO: make compute_attention python callable
    os.system(
        f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset stcmds --dataset_metafile metadata.csv --data_path ./recipes/st-cmds/st-cmds/  --use_cuda true"
    )

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init the model
model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
