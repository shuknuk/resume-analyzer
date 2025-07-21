# main.py
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
    "https://resume-analyzer-frontend-seven.vercel.app",
    "https://ranalyzer.vercel.app"
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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
tools = [TavilySearchResults(max_results=3)]

prompt = PromptTemplate.from_template("""
You are an expert career coach and professional resume strategist. Your goal is to provide a detailed, objective, and constructive analysis of a resume, delivered in a structured JSON format.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a valid JSON object that follows this exact structure:
{{
  "score": <an integer score from 0 to 100 representing the resume's overall quality and fit, if context is provided>,
  "scoreRationale": "<A brief, one-sentence explanation for why you gave that score>",
  "strengths": [<an array of 2-4 strings, where each string is a specific, actionable strength>],
  "improvements": [
    {{
      "suggestion": "<A concise heading for the improvement area, e.g., 'Quantify Achievements'>",
      "explanation": "<A one-sentence explanation of WHY this improvement is important>",
      "example": "<A concrete 'before' and 'after' example. e.g., 'Instead of: Managed a project. Try: Managed a 3-month project with a $5k budget, resulting in a 15% increase in user engagement.'>"
    }}
  ]
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
    handle_parsing_errors=True,
    return_intermediate_steps=True # This is the key to getting the log!
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
        # Build the input text dynamically
        input_text = f"Analyze this resume:\n---{data.resume_text}\n---"
        if data.job_description and data.job_description.strip():
            input_text += f"\nTailor the analysis for this specific job description:\n---{data.job_description}\n---"
        if data.company_name and data.company_name.strip():
            input_text += f"\nAlso, provide extra insights by researching this company:\n---{data.company_name}\n---"

        response = await agent_executor.ainvoke({"input": input_text})
        
        # Extract the final answer
        analysis_data = _parse_json_from_string(response['output'])
        
        # Format the intermediate steps into a clean log string
        log_string = ""
        for step in response.get('intermediate_steps', []):
            try:
                action, observation = step
                # Handle both AgentAction objects and other formats
                if hasattr(action, 'log'):
                    log_string += f"Thought: {action.log.strip()}\n"
                if hasattr(action, 'tool'):
                    log_string += f"Action: {action.tool}\n"
                elif hasattr(action, 'name'):
                    log_string += f"Action: {action.name}\n"
                if hasattr(action, 'tool_input'):
                    log_string += f"Action Input: {action.tool_input}\n"
                log_string += f"Observation: {observation}\n\n"
            except Exception as log_error:
                print(f"Error processing log step: {log_error}")
                log_string += f"Step: {step}\n\n"

        # Return both the analysis and the log
        return {"analysis": analysis_data, "log": log_string}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")
