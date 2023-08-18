import os
import sys
from time import time
from typing import Any

from datasets import load_dataset, Audio, Dataset
from evaluate import AutomaticSpeechRecognitionEvaluator
from huggingface_hub import login
from numpy import ndarray
from transformers import AutomaticSpeechRecognitionPipeline, pipeline


def auth(filename: str) -> None:
    with open(filename) as f:
        token = f.read().strip()
        login(token, add_to_git_credential=False)


def prepare_pipeline(model_id: str, lang_id: str) -> None:
    global pipe
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
    )
    pipe.model.load_adapter(lang_id)
    pipe.tokenizer.set_target_lang(lang_id)


def prepare_dataset(lang: str, split: str = "test") -> None:
    global dataset
    path = "mozilla-foundation/common_voice_13_0"
    dataset = load_dataset(path, name=lang, split=split)
    datatype = Audio(sampling_rate=16000)
    dataset = dataset.cast_column("audio", datatype)


def get_data_sample(sample_index: int = 0) -> ndarray:
    global dataset
    for index, data_item in enumerate(dataset):
        if index == sample_index:
            return data_item["audio"]["array"]


def get_transcription(sample: ndarray) -> str:
    output = pipe(sample)
    return output["text"]


def get_evaluation() -> Any:
    evaluator = AutomaticSpeechRecognitionEvaluator()
    evaluator.PIPELINE_KWARGS.pop("truncation")  # Грязный хак
    result = evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        metric="wer",
        input_column="audio",
    )
    return result["wer"]


def main() -> None:
    before = time()
    filename = "token.txt"
    auth(filename)
    print(f"Аутентификация в Hugging Face: {time() - before:.2f} с.")

    before = time()
    # Базовая модель с 300 млн параметров для дообучения
    # model_id = "facebook/mms-300m"
    # Базовая модель с 1 млрд параметров для дообучения
    # model_id = "facebook/mms-1b"
    # Готовая модель с 1 млрд параметров и поддерживающая 1162 языка
    model_id = "facebook/mms-1b-all"
    model_lang = "sah"

    prepare_pipeline(model_id, model_lang)
    print(f"Загрузка модели: {time() - before:.2f} с.")

    before = time()
    dataset_lang = "sah"
    prepare_dataset(dataset_lang, "test[:10%]")
    sample = get_data_sample()
    print(f"Загрузка набора данных: {time() - before:.2f} с.")

    before = time()
    transcription = get_transcription(sample)
    print(transcription)
    duration = time() - before
    print(f"Вывод результата: {duration:.2f} с.")

    print(f"Прогноз длительности расчета WER: {len(dataset) * duration} с.")
    before = time()
    evaluation = get_evaluation()
    print(f"WER: {evaluation}")
    print(f"Расчет WER: {time() - before:.2f} с.")


def memory_usage():
    pid = os.getpid()
    with open(f"/proc/{pid}/status") as f:
        for line in f.readlines():
            if line.startswith("VmPeak:"):
                _, result = line.strip().split(maxsplit=1)
                print(f"Memory consumption: {result}")


if __name__ == "__main__":
    pipe: AutomaticSpeechRecognitionPipeline
    dataset: Dataset

    main()

    if sys.platform == "linux":
        memory_usage()
