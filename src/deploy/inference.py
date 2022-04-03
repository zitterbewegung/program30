import requests
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline, Conversation
from transformers import (
    TFAutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    BlenderbotTokenizer,
    BlenderbotSmallTokenizer,
    BlenderbotForConditionalGeneration,
    Conversation,
)
from PIL import Image
import requests
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM ,AutoModelForCausalLM, pipeline, Conversation
import logging, json
from typing import List, Any, Tuple, Dict
import os
#from GPTJ.Basic_api import SimpleCompletion
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from functools import lru_cache

HFAUTH = os.getenv('HEADERS')
headers = {"Authorization": HFAUTH }

@lru_cache
def smalltalk(utterance:str):
    nlp = pipeline("text2text-generation", model='facebook/blenderbot-1B-distill', device=1)
    return nlp(utterance)[0].get('generated_text')

def emotion_category(utterance: str) -> Dict[str, str]:
    '''
    From https://akoksal.com/articles/zero-shot-text-classification-evaluation
    
    '''
    candidate_labels = ["anger", "fear", "joy", "love", "sadness", "surprise"]
    classifier = pipeline("zero-shot-classification", device=1)
    return  classifier(utterance, candidate_labels)



def compute_sentiment(utterance: str) -> Dict[str, str]:
    nlp = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=0)
    result = nlp(utterance)[0]['label']
    return result

def query_huggingface(payload):
	API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot_small-90M"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def infer_sentiment_huggingface(payload):
	API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def infer_simple_situation(prompt: str) -> str:
#     temperature   = 0.7
#     top_probability = 1.0
#     max_length    = 20
#     #context = "This is a girl that is nice that wants to go on a date."

#     # examples = {
#     # "5 + 5": "10",
#     # "6 - 2": "4",
#     # "4 * 15": "60",
#     # "10 / 5": "2",
#     # "144 / 24": "6",
#     # "7 + 1": "8"}

    
    #context_setting = Completion(context, examples)
    #query = SimpleCompletion(prompt, length=max_length, t=temperature, top=top_probability)
    #Query = query.simple_completion()
    return 'Test'

def analysis_and_conversation(utterance):
    nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")							   
    user_sentiment_result = nlp(utterance)[0]["label"]			      
    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-1B-distill").to('cuda:0')
    inputs = tokenizer([utterance], return_tensors="pt")								   
    reply_ids = model.generate(**inputs)												   
    responses = [																		   
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
        for g in reply_ids															   
    ]																					   
    response_utterance_sentiments = [nlp(x) for x in responses]						   
    response_event = {																   
        'utterance_agent'  :  responses,												
        'sentiment_agent'  :  response_utterance_sentiments,		   
        'utterance_user'   :  utterance,												
        'sentiment_user'   :  nlp(utterance)[0]["label"],						           																				   
    }
    return response_event
    

