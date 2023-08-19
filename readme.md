# Telegram-бот для распознавания речи на якутском языке

## Запуск бота

1. Установить пакеты
   ```commandline   
   pip install -r requirements.txt
   ```
2. Скачать модель
   ```commandline
   python prepare.py
   ```
3. Установить переменную окружения `TG_TOKEN`
   ```commandline
   set TG_TOKEN="токен"
   ```
4. Запустить бота
   ```commandline
   python bot.py
   ```

## Текущий результат

В данный момент используется готовая модель [mms-1b-all](https://huggingface.co/facebook/mms-1b) обученная на 1162 языка.

Модель занимает 13 ± 0.5 Гб в ОЗУ.

Набор данных: Mozilla CommonVoice 13.0 для якутского языка.

CER: 0.067, WER: 0.321. 
