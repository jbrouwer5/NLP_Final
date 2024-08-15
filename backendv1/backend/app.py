from flask import Flask, request, jsonify
from transcribe.speech_to_text import transcribe
from translate.translation import translate_en_to_fr
from transformers import MarianMTModel, MarianTokenizer
import os
from flask_cors import CORS
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging

logging.basicConfig(filename='app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger(__name__)

# FOR TRANSCRIPTION
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-tiny"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# FOR TRANSLATION
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        audio = request.files['file']
        logger.error("Retreived the file")
    except:
        logger.error("Couldn't find the audio file")
        return jsonify({'error': 'No audio file provided'}), 400
    
    if audio.filename == '':
        logger.error('No Selected Audio File')
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the audio file to a temporary location
        audio_file_path = os.path.join('/tmp', audio.filename)
        audio.save(audio_file_path)
        logger.error('Saved the Audio File')
    except:
        logger.error("Error while saving audio file")
        return jsonify({'error': 'Error while saving audio file'}), 400
    
    try:
        # Process the audio file
        print("starting transcription")
        transcribed_text = transcribe(pipe, audio_file_path)
        print("done with transcription")
        print("starting translation")
        translated_text = translate_en_to_fr(tokenizer, model, transcribed_text)
        print("done with translation")
    finally:
        # Clean up the saved file
        os.remove(audio_file_path)

    return jsonify({'text': translated_text, 'statusCode' : 200})

if __name__ == '__main__':
    app.run(debug=True)
