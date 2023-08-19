from os import getenv
from tempfile import NamedTemporaryFile

from librosa import load
from telebot import TeleBot
from telebot.types import Message
from transformers import pipeline

bot = TeleBot(getenv("TG_TOKEN"))


# Помощь
@bot.message_handler(commands=["start", "help"])
def send_welcome(message: Message):
    text = "Howdy, how are you doing?"
    bot.reply_to(message, text)


@bot.message_handler(content_types=["audio", "voice"])
def echo_all(msg: Message):
    # Скачивание записи
    record = msg.audio or msg.voice
    handle = bot.get_file(record.file_id)
    content = bot.download_file(handle.file_path)
    with NamedTemporaryFile() as new_file:
        new_file.write(content)
        audio, _ = load(new_file.name, sr=16_000)

    # Распознавание речи
    transcript = pipe(audio)

    # Ответ бота
    bot.reply_to(msg, transcript["text"])


# Загрузка модели и адаптера
pipe = pipeline(
    task="automatic-speech-recognition",
    model="facebook/mms-1b-all",
)
pipe.model.load_adapter("sah")
pipe.tokenizer.set_target_lang("sah")

# TODO: надо использовать веб-хуки
bot.infinity_polling()
