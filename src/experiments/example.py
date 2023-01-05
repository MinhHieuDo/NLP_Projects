from transformers import pipeline
translator = pipeline("translation_en_to_fr")
text_en = "It is very easy to use transformers"
text_fr = translator(text_en,max_length=40)
print(text_en)
print(text_fr)