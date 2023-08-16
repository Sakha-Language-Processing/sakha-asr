"""
Проверка работы модели
"""

import os
import sys
from time import time
from typing import Iterable

import evaluate
import torch
from datasets import load_dataset, Audio, DatasetDict
from huggingface_hub import login
from numpy import ndarray
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor


def auth(filename: str) -> None:
    """
    Аутентификация
    """
    with open(filename) as f:
        token = f.read().strip()
        login(token)


def get_processor(model_id: str, lang_id: str) -> Wav2Vec2Processor:
    """
    Обработчик данных
    """
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.set_target_lang(lang_id)
    return processor


def get_model(model_id: str, lang_id: str) -> Wav2Vec2ForCTC:
    """
    Создать модель
    """
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.load_adapter(lang_id)
    return model


def get_data_sample(lang: str, sample_index: int = 0) -> ndarray:
    """
    Получить образец данных
    """
    path = "common_voice_13_0"
    dataset = load_dataset(path, lang, split="test", streaming=True)
    datatype = Audio(sampling_rate=16000)
    data = dataset.cast_column("audio", datatype)
    sample = None
    for index, data_item in enumerate(data):
        if index == sample_index:
            sample = data_item["audio"]["array"]
            break
    return sample


def get_prediction(processor: Wav2Vec2Processor,
                   model: Wav2Vec2ForCTC,
                   sample: ndarray) -> str:
    """
    Получить предсказание
    """

    inputs = processor(sample, sampling_rate=16_000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    return transcription


def main() -> None:
    """
    Точка входа
    """

    before = time()
    filename = "token.txt"
    auth(filename)
    print(f"Аутентификация в Hugging Face: {time() - before:.2f} s")

    before = time()
    # Базовая модель с 300 млн параметров для дообучения
    # model_id = "facebook/mms-300m"
    # Базовая модель с 1 млрд параметров для дообучения
    # model_id = "facebook/mms-1b"
    # Готовая модель с 1 млрд параметров и поддерживающая 1162 языка
    model_id = "facebook/mms-1b-all"
    model_lang = "sah"
    processor = get_processor(model_id, model_lang)
    model = get_model(model_id, model_lang)
    print(f"Загрузка модели и обработчика: {time() - before:.2f} s")

    before = time()
    sample_lang = "sah"
    sample = get_data_sample(sample_lang)
    print(f"Загрузка образца данных: {time() - before:.2f} s")

    before = time()
    prediction = get_prediction(processor, model, sample)
    print(prediction)
    print(f"Вывод результата: {time() - before:.2f} s")


def print_memory_usage():
    pid = os.getpid()
    with open(f"/proc/{pid}/status") as f:
        for line in f.readlines():
            if line.startswith("VmPeak:"):
                _, result = line.strip().split(maxsplit=1)
                print(f"Memory consumption: {result}")


if __name__ == '__main__':
    main()
    if sys.platform == "linux":
        print_memory_usage()
