from fastapi import FastAPI, File, UploadFile
from phidata.agent import Agent, RunResponse
from phidata.tools.search import DuckDuckGoSearch
from PyPDF2 import PdfReader
from pydantic import BaseModel
from phi.model.huggingface import HuggingFaceChat
import uvicorn

app = FastAPI()

class UserInfo(BaseModel):
    name: str
    field: str
    experience: str
    academics: str
    interests: str
    skills: str

user_info = None

# Career Advice Agent
career_agent = Agent(model=HuggingFaceChat(
        id="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
        max_tokens=4096),
        tools=[DuckDuckGoSearch()],
        markdown=True)

# Resume Review Agent
resume_agent = Agent(model=HuggingFaceChat(
        id="NousResearch/DeepHermes-3-Llama-3-8B-Previewt",
        max_tokens=4096),
        markdown=True)

@app.get("/get_user_info")
def get_user_info(info: UserInfo):
    global user_info
    user_info = info
    return {"message": "User info updated", "user_info": user_info.dict()}

@app.post("/predict/career")
def predict_career():
    if not user_info:
        return {"error": "User info not provided"}
    
    query = (
        f"Analyze academic performance, interests, skills, and market trends for: {user_info.dict()}. "
        f"Suggest suitable career paths matching current industry trends. "
        f"Recommend relevant courses, certifications, and projects to improve employability."
    )
    result = career_agent.run(query)
    return {"career_advice": result}

@app.post("/predict/resume")
def predict_resume(file: UploadFile = File(...)):
    pdf_reader = PdfReader(file.file)
    resume_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    
    query = f"Review this resume and suggest improvements: {resume_text[:1000]}..."
    result = resume_agent.run(query)
    return {"resume_insights": result}


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port="8080")
