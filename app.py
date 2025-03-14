from transformers import pipeline
from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response


def predict(content:str):
    messages = [
        {"role": "user", "content": content},
    ]
    pipe = pipeline("text-generation", model="NousResearch/DeepHermes-3-Llama-3-8B-Preview")
    res = pipe(messages)
    print(res)
    return res 

app = FastAPI()
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_route(text):
    try:
        text = predict(text)
        return text
    except Exception as e: 
        raise e
    
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)