#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:58:11 2024

@author: aman
"""
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import whisper

model = whisper.load_model('base')
result = model.transcribe('song1.wav', fp16 = False)
txt = result['text']

tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

emotion_labels = emotion(txt)

print(emotion_labels)

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

model_outputs = classifier(txt)

print(model_outputs[0])

#print(txt)