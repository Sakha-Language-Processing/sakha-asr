from transformers import pipeline

pipe = pipeline(
    task="automatic-speech-recognition",
    model="facebook/mms-1b-all",
)
pipe.model.load_adapter("sah")
pipe.tokenizer.set_target_lang("sah")
