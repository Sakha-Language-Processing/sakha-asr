# Дообучение модели mms-300m на якутский язык

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

## Использование модели в 

```python
from transformers import AutoProcessor, AutoModelForPreTraining

processor = AutoProcessor.from_pretrained("facebook/mms-300m")
model = AutoModelForPreTraining.from_pretrained("facebook/mms-300m")
```

## Ссылки

* https://github.com/huggingface/transformers/tree/main/examples/pytorch/speech-recognition#connectionist-temporal-classification-with-adapters
* https://huggingface.co/blog/mms_adapters
