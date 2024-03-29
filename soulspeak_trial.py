import streamlit as st
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import whisper

st.title('Emotion Detection from Audio')

model = whisper.load_model('base')
#result = model.transcribe('song1.wav', fp16 = False)
#txt = result['text']

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

#emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#emotion_labels = emotion(txt)

#print(emotion_labels)

#classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

#model_outputs = classifier(txt)

#print(model_outputs[0])

audio_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])
if audio_file is not None:
    result = model.transcribe(audio_file, fp16=False) 
    txt = result['text']
    
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    emotion_labels = emotion(txt)

    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    model_outputs = classifier(txt)

    st.write("Transcribed Text:", txt)
    st.write("Emotion Labels:", emotion_labels)
    st.write("Emotion Classification:", model_outputs[0])
