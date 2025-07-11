import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

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
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# --- Pydantic Models ---
class ResumeData(BaseModel):
    resume_text: str
    job_description: str | None = None
    company_name: str | None = None


# --- LangChain Agent Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)
tools = [TavilySearchResults(max_results=3)]

# --- THE FIX IS HERE: Using the full, correct prompt template ---
prompt = PromptTemplate.from_template("""
You are an expert career coach and professional resume reviewer for tech roles.
Your goal is to provide a structured JSON response to help the user improve their resume.

You have access to the following tools:
{tools}

To get your final answer, you must use the following format:

Question: the user's request, including the resume and job description.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: a valid JSON object that follows this exact structure:
{{
  "score": <an integer score from 0 to 100>,
  "summary": "<a one-sentence summary>",
  "keywords": [<an array of 5-7 relevant keywords>],
  "strengths": [<an array of strings>],
  "improvements": [<an array of strings>]
}}

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

def _parse_json_from_string(text: str) -> dict:
    try:
        start_index = text.find('{')
        end_index = text.rfind('}') + 1
        if start_index == -1 or end_index == 0:
            raise ValueError("No JSON object found in the string.")
        json_str = text[start_index:end_index]
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON from agent response: {e}")
        raise e


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "LangChain Resume Agent API is running."}

@app.post("/analyze")
async def analyze_resume(data: ResumeData):
    try:
        input_text = f"Resume:\n{data.resume_text}\n"
        if data.job_description:
            input_text += f"\nJob Description:\n{data.job_description}\n"
        if data.company_name:
            input_text += f"\nCompany to research:\n{data.company_name}\n"

        response = await agent_executor.ainvoke({"input": input_text})
        
        analysis_data = _parse_json_from_string(response['output'])

        return {"analysis": analysis_data}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")