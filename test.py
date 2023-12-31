import os
import re
import sys
from time import time

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

    def rm_spec_chars(batch):
        col = "sentence"
        chars = "[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']"
        batch[col] = re.sub(chars, "", batch[col]).lower()
        return batch

    global dataset
    path = "mozilla-foundation/common_voice_13_0"
    datatype = Audio(sampling_rate=16000)
    dataset = load_dataset(path, name=lang, split=split)
    dataset = dataset.cast_column("audio", datatype)
    dataset = dataset.map(rm_spec_chars)


def get_data_sample(sample_index: int = 0) -> ndarray:
    global dataset
    for index, data_item in enumerate(dataset):
        if index == sample_index:
            # print(data_item["sentence"])
            return data_item["audio"]["array"]


def get_transcription(sample: ndarray) -> str:
    output = pipe(sample)
    return output["text"]


def get_evaluation() -> (float, float):
    evaluator = AutomaticSpeechRecognitionEvaluator()
    evaluator.PIPELINE_KWARGS.pop("truncation")  # Грязный хак
    result_cer = evaluator.compute(
        model_or_pipeline=pipe, data=dataset,
        metric="cer", input_column="audio",
    )
    result_wer = evaluator.compute(
        model_or_pipeline=pipe, data=dataset,
        metric="wer", input_column="audio",
    )
    return result_cer["cer"], result_wer["wer"]


def main() -> None:
    before = time()
    filename = "token.txt"
    auth(filename)
    print(f"Аутентификация в Hugging Face: {time() - before:.3f} с.")

    before = time()
    model_id = "facebook/mms-1b-all"
    model_lang = "sah"

    prepare_pipeline(model_id, model_lang)
    print(f"Загрузка модели: {time() - before:.3f} с.")

    before = time()
    dataset_lang = "sah"
    prepare_dataset(dataset_lang, "test")
    sample = get_data_sample()
    print(f"Загрузка набора данных: {time() - before:.3f} с.")

    before = time()
    transcription = get_transcription(sample)
    print(transcription)
    duration = time() - before
    print(f"Вывод результата: {duration:.3f} с.")

    print(f"Прогноз времени расчета WER: {len(dataset) * duration * 2:.3f} с.")
    before = time()
    cer, wer = get_evaluation()
    print(f"CER: {cer:.3f}, WER: {wer:.3f}")
    print(f"Расчет WER: {time() - before:.3f} с.")


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
