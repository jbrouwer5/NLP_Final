from transformers import MarianMTModel, MarianTokenizer

def translate_en_to_fr(tokenizer, model, sentence):

    tokens = tokenizer.encode(sentence, return_tensors='pt')
    
    translated_tokens = model.generate(tokens)
    
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_sentence

if __name__ == "__main__":
    english_sentence = "This is a test sentence."
    french_translation = translate_en_to_fr(english_sentence)
    print(french_translation)
