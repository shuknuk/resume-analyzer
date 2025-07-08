import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# --- App Configuration ---
app = FastAPI()

# --- CORS Configuration ---
origins = [
    "http://localhost:3000",
    "https://resume-analyzer-frontend-seven.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key Configuration ---
try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Google API Key not found.")
    genai.configure(api_key=google_api_key)
except ValueError as e:
    print(e)

# --- Pydantic Models ---
class ResumeData(BaseModel):
    resume_text: str
    job_description: str | None = None # <-- UPDATE THIS

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Resume Analyzer API is running."}

@app.post("/analyze")
async def analyze_resume(data: ResumeData):
    try:
        generation_config = {"response_mime_type": "application/json"}
        model = genai.GenerativeModel('gemini-2.5-pro', generation_config=generation_config)

        # model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config=generation_config)
        

        # --- UPDATE THIS WHOLE SECTION ---
        # Start with the base prompt
        prompt_template = """
        You are an expert career coach and professional resume reviewer for tech roles.
        Analyze the following resume text and respond with ONLY a valid JSON object.
        The JSON object must follow this exact structure:
        {{
          "score": <an integer score from 0 to 100 representing the resume's quality>,
          "summary": "<a one-sentence summary of the overall feedback>",
          "keywords": [<an array of 5-7 relevant keywords found or missing from the resume>],
          "strengths": [<an array of strings, where each string is a specific, actionable strength>],
          "improvements": [<an array of strings, where each string is a specific, actionable area for improvement>]
        }}
        """

        # If a job description is provided, add comparison instructions to the prompt
        if data.job_description and data.job_description.strip():
            prompt_template += f"""
            Crucially, you must tailor your analysis by comparing the resume against the following job description.
            The score should reflect how well the resume is tailored for this specific job.
            The keywords, strengths, and improvements should all be in the context of this job description.

            JOB DESCRIPTION:
            ---
            {data.job_description}
            ---
            """

        # Add the resume text to the final prompt
        prompt = prompt_template + f"""
        Here is the resume text to analyze:
        ---
        {data.resume_text}
        ---
        """

        response = await model.generate_content_async(prompt)
        
        analysis_data = json.loads(response.text)
        
        return {"analysis": analysis_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")