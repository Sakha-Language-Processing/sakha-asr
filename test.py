import torch
from datasets import load_dataset, Audio
from huggingface_hub import login
from numpy import ndarray
from transformers import AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor


def auth() -> None:
    """
    Аутентификация
    """
    with open("token.txt") as f:
        token = f.read().strip()
        login(token)


def get_model(model_id: str) -> (Wav2Vec2Processor, Wav2Vec2ForCTC):
    """
    Создать модель
    """
    lang_id = "sah"
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.set_target_lang(lang_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.load_adapter(lang_id)
    return processor, model


def get_sample(lang: str) -> ndarray:
    """
    Получить образец данных
    """
    path = "common_voice_13_0"
    dataset = load_dataset(path, lang, split="test", streaming=True)
    datatype = Audio(sampling_rate=16000)
    data = dataset.cast_column("audio", datatype)
    sample = None
    for i, v in enumerate(data):
        if i == 1:
            sample = v["audio"]["array"]
            break
    return sample


def get_prediction(sample: ndarray) -> str:
    """
    Получить предсказание
    """

    # Базовая модель с 300 млн параметров для дообучения
    # model_id = "facebook/mms-300m"
    # Базовая модель с 1 млрд параметров для дообучения
    # model_id = "facebook/mms-1b"
    # Готовая модель с 1 млрд параметров и поддерживающая 1162 языка
    model_id = "facebook/mms-1b-all"
    processor, model = get_model(model_id)

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
    auth()
    sample = get_sample("sah")
    prediction = get_prediction(sample)
    print(prediction)


if __name__ == '__main__':
    main()
