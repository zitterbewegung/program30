import os
import pickle
import numpy as np
import PIL.Image
from pydantic import BaseModel, validator
from uuid import UUID, uuid4
from typing import Optional, List, Dict
import enum, random
from fastapi import FastAPI
from datetime import datetime
from typing import List, Optional
import uuid
import logic
import requests
import urllib.parse
import uvicorn
import inference
import torch
from transformers import pipeline, Conversation
from functools import lru_cache
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
    Conversation,
    pipeline,
    set_seed,
    Conversation
)
import secrets
#from dalle_mini.model import CustomFlaxBartForConditionalGeneration
from transformers import BartTokenizer
#import jax
import random
#from tqdm.notebook import tqdm, traneg
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


set_seed(42)

NLP_SENTIMENT = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0,
)

NLP_CONVERSATION_BLENDER = pipeline(
    "text2text-generation",
    model="facebook/blenderbot-1B-distill",
    device=0,
)


# NLP_CONVERSATION_C1 = pipeline(
#     "text2text-generation",
#     model="iokru/c1-1.3B",
#     device=1,
# )
#NLP_CONVERSATION_C1_NEO = pipeline('text-generation', model='iokru/c1-1.3B', device=1)
#NLP_CONVERSATION_C1_NEO = pipeline('text-generation', model='iokru/c1-1.3B', device=1)


# make sure we use compatible versions
#DALLE_REPO = "flax-community/dalle-mini"
#DALLE_COMMIT_ID = "4d34126d0df8bc4a692ae933e3b902a1fa8b6114"

# set up tokenizer and model
#NLP_IMG_DALLE_TOKENIZER = BartTokenizer.from_pretrained(DALLE_REPO, revision=DALLE_COMMIT_ID)
#NLP_IMG_DALLE_MODEL = CustomFlaxBartForConditionalGeneration.from_pretrained(
#    DALLE_REPO, revision=DALLE_COMMIT_ID)
#NLP_IMG_DALLE_MODEL     = AutoModelForSeq2SeqLM.from_pretrained("flax-community/dalle-mini")


NLP_SENTIMENT_MULTI_CLASS = pipeline("zero-shot-classification", multi_labe=True, device=0)

#NLP_SITUATION = pipeline('text-generation', model='addy88/gpt-j-8bit', device=1)

# NLP_SENTIMENT = pipeline(
#     "sentiment-analysis",
#     model="distilbert-base-uncased-finetuned-sst-2-english",
#     device=1,
# )


# NLP_SENTIMENT_MULTI_CLASS = pipeline("zero-shot-classification", multi_class=True)


@app.get("/")
async def read_root():

    return {"Ara": torch.cuda.is_available()}

def fake_decode_token(token):
    return User(
        username=token + "fakedecoded", email="john@example.com", full_name="John Doe"
    )

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

def smalltalk(utterance: str):
    return NLP_CONVERSATION_BLENDER(utterance)[0].get("generated_text")

#def C1_talk(utterance: str):
#    return NLP_CONVERSATION_C1_NEO(utterance)[0].get("generated_text").split('\n')[1]


def compute_sentiment(utterance: str) -> Dict[str, str]:
    result = NLP_SENTIMENT(utterance)[0]["label"]
    return result

def compute_multi_class_sentiment(utterance: str) -> Dict[str,str]:
    result =  NLP_SENTIMENT_MULTI_CLASS("zero-shot-classification", candidate_labels =["anger", "fear", "joy", "love", "sadness", "surprise"],  multi_class=True)
    return result

@app.post("/conversation_c1")
async def create_conversation_c1(username: str, utterance: str, token: str = Depends(oauth2_scheme)) -> Event:
    response = C1_talk(utterance).replace('Deleted User: ', '')
    user_sentiment = compute_sentiment(utterance)
    agent_sentiment = compute_sentiment(utterance)

    return logic.answer_question(
        utterance,
        response,
        user_sentiment,
        agent_sentiment,
        username,
        "visual_novel",
        "japan",
    )
@app.post("/conversation")
async def create_conversation(username: str, utterance: str) -> Event:
    response = smalltalk(utterance)
    user_sentiment = compute_sentiment(utterance)
    agent_sentiment = compute_sentiment(utterance)

    return logic.answer_question(
        utterance,
        response,
        user_sentiment,
        agent_sentiment,
        username,
        "visual_novel",
        "japan",
    
)

@app.get("/face")
async def generate_face(tags: str):
    tflib.init_tf()
    _G, _D, Gs = pickle.load(
        open("results/02051-sgan-faces-2gpu/network-snapshot-021980.pkl", "rb")
    )
    Gs.print_layers()

    for i in range(0, 10):
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(
            latents,
            None,
            truncation_psi=0.6,
            randomize_noise=True,
            output_transform=fmt,
        )
        os.makedirs(con2fig.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, "example-" + str(i) + ".png")
        PIL.Image.fromarray(images[0], "RGB").save(png_filename)
    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".png", delete=False) as FOUT:
        FOUT.write(images[0])
    return FileResponse(FOUT.name, media_type="image/png")
    # return FileResponse("image.jpeg")

    # smalltalk(utterance)

@app.get("/situation")
async def situation(username: str, prompt: str) -> str:

     model = AutoModelForCausalLM.from_pretrained('hakurei/lit-6B')
     tokenizer = AutoTokenizer.from_pretrained('hakurei/lit-6B')

     input_ids = tokenizer.encode(prompt, return_tensors='pt')
     output = model.generate(input_ids, do_sample=True, temperature=1.0, top_p=0.9, repetition_penalty=1.2, max_length=len(input_ids[0])+100, pad_token_id=tokenizer.eos_token_id)

     generated_text = tokenizer.decode(output[0])
     return generated_text
	
@app.post("/analyze_sentiment")
async def emotion_category(utterance: str) -> Dict[str, str]:
    """
    From https://akoksal.com/articles/zero-shot-text-classification-evaluation

    """
    return compute_multi_class_sentiment(utterance)
