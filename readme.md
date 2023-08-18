# Дообучение модели Facebook MMS на якутский язык

## Основы

https://github.com/facebookresearch/fairseq — Facebook AI Research Sequence-to-Sequence Toolkit written in Python.

https://github.com/facebookresearch/fairseq/tree/main/examples/mms — The Massively Multilingual Speech (MMS): Scaling Speech Technology to 1000+ languages.

Базовые модели для дообучения:
* [Модель](https://dl.fbaipublicfiles.com/mms/pretraining/base_300m.pt), [HuggingFace](https://huggingface.co/facebook/mms-300m) — 300 млн параметров.
* [Модель](https://dl.fbaipublicfiles.com/mms/pretraining/base_1b.pt), [HuggingFace](https://huggingface.co/facebook/mms-1b) — 1 млрд параметров.

## Подходы к дообучению

[Скрипт для дообучения слоев адаптера](https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition#connectionist-temporal-classification-with-adapters)

[Статья в блоге](https://huggingface.co/blog/mms_adapters)

[Старый пример](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#fine-tune-a-pre-trained-model-with-ctc)

## Полезные команды

Перекодирование на частоту дискретизации 16 кгц:
```sh
ffmpeg -y -i ./audio_samples/audio.mp3 -ar 16000 ./audio_samples/audio.wav
```

Запуск распознавания
```sh
python examples/mms/asr/infer/mms_infer.py \
    --model "../mms1b_fl102.pt" \
    --lang "eng" \
    --audio "./audio_samples/audio.wav"
```

## Распознавание образцов

```python
from datasets import Audio, load_dataset
from transformers import pipeline

data_path = "mozilla-foundation/common_voice_13_0"
data_lang = "sah"
dataset = load_dataset(data_path, name=data_lang, split="test")
datatype = Audio(sampling_rate=16000)
dataset = dataset.cast_column("audio", datatype)
sample = next(iter(dataset))

model_id = "facebook/mms-1b-all"
pipe = pipeline(task="automatic-speech-recognition", model=model_id)
lang_id = "sah"
pipe.model.load_adapter(lang_id)
pipe.tokenizer.set_target_lang(lang_id)
transcript = pipe(sample)

print(transcript)
```

## Текущий результат

В данный момент используется готовая модель MMS-1B-all обученная на 1162 языка.

Модель занимает 13 ± 0.5 Гб в ОЗУ.

WER тестового набора CommonVoice 13.0 для якутского языка: 0.47. 

## Ссылки

* https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition#connectionist-temporal-classification-with-adapters
* https://huggingface.co/blog/mms_adapters
