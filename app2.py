from fastapi import FastAPI, File, UploadFile
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from PyPDF2 import PdfReader
from pydantic import BaseModel
from phi.model.huggingface import HuggingFaceChat
import uvicorn
import os

app = FastAPI()

class UserInfo(BaseModel):
    name: str
    field: str
    experience: str
    academics: str
    interests: str
    skills: str

career_agent = Agent(
    model=HuggingFaceChat(id="meta-llama/Meta-Llama-3-8B-Instruct", max_tokens=4096),
    tools=[DuckDuckGo()],
    markdown=True
)

resume_agent = Agent(
    model=HuggingFaceChat(id="meta-llama/Meta-Llama-3-8B-Instruct", max_tokens=4096),
    markdown=True
)

@app.get("/", tags=["authentication"])
async def index():
    return {"message": "Welcome to the API. Visit /docs for API documentation."}

@app.post("/predict/career")
def predict_career(user_info: UserInfo):
    query = (
        f"Analyze academic performance, interests, skills, and market trends for: {user_info.model_dump()}. "
        f"Use a web search to find the latest industry trends. "
        f"Suggest suitable career paths matching industry trends. "
        f"Find relevant courses, certifications, and projects from the internet using DuckDuckGo and provide links."
    )

    try:
        result = career_agent.run(query, tool_choice="auto")

        if hasattr(result, "tool_calls") and result.tool_calls:
            result.tool_calls = [t.__dict__ for t in result.tool_calls]

        return {"career_advice": result.__dict__}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/advice")
def predict_advice(user_info: UserInfo, text: str):
    query = (
        f"Depending on the following user data, provide friendly and cheerful advice to them, also provide any relevant online resources from youtube, wikipedia, research paper articles through DuckDuckGo search: {user_info.model_dump()}. "
        f"user query: {text}"
    )

    try:
        result = career_agent.run(query, tool_choice="auto")

        if hasattr(result, "tool_calls") and result.tool_calls:
            result.tool_calls = [t.__dict__ for t in result.tool_calls]

        return {"emotional_advice": result.__dict__}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/resume")
def predict_resume(file: UploadFile = File(...)):
    try:
        pdf_reader = PdfReader(file.file)
        resume_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        query = f"Review this resume and suggest improvements: {resume_text[:1000]}..."
        result = resume_agent.run(query)

        return {"resume_insights": result.__dict__}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
