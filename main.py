import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware

# Load environment variables
load_dotenv()

# --- App Configuration ---
app = FastAPI()

# --- CORS Configuration ---
# Define the list of "origins" (websites) that are allowed to make requests
origins = [
    "http://localhost:3000", # The address of our Next.js frontend
    "https://resume-analyzer-frontend-seven.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)


# --- API Key Configuration ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Google API Key not found. Please set it in the .env file.")
    genai.configure(api_key=google_api_key)
except ValueError as e:
    print(e)


# --- Pydantic Models ---
class ResumeData(BaseModel):
    resume_text: str


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Resume Analyzer API is running."}

@app.post("/analyze")
async def analyze_resume(data: ResumeData):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        You are an expert career coach and professional resume reviewer for tech roles.
        Analyze the following resume text and provide clear, actionable feedback.
        Structure your feedback into three sections using markdown headings:

        ### Strengths
        What this resume does well. Be specific and encouraging.

        ### Areas for Improvement
        What can be improved. Be constructive and provide clear examples.
        Focus on action verbs, quantifiable achievements, and clarity.

        ### Formatting Suggestions
        Tips on layout, whitespace, and readability.

        Here is the resume text:
        ---
        {data.resume_text}
        ---
        """

        response = await model.generate_content_async(prompt)

        return {"analysis": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")