from transformers import pipeline
from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import torch

pipe = pipeline(
    "text-generation", 
    model="NousResearch/DeepHermes-3-Llama-3-8B-Preview", 
    device_map="auto",  # Automatically selects GPU if available
    torch_dtype=torch.bfloat16
)

def predict(content: str):
    res = pipe(content, max_length=512, do_sample=True)  # Ensure proper input format
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
    uvicorn.run(app, host="172.29.128.1", port="8080")