import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# --- New LangChain Imports ---
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
# We no longer need to configure genai directly, LangChain handles it.
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# --- Pydantic Models (Data Shape) ---
class ResumeData(BaseModel):
    resume_text: str
    job_description: str | None = None
    company_name: str | None = None # New field for the company name


# --- LangChain Agent Setup ---

# 1. The LLM (The "Brain")
# We initialize our Gemini model through the LangChain adapter.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5)

# 2. The Tools (The "Hands")
# We define the tools the agent can use. For now, just one: Tavily web search.
tools = [TavilySearchResults(max_results=3)]

# 3. The Prompt (The "Instructions")
# This is a special prompt template that tells the agent how to reason and use tools.
# The 'react' framework here stands for "Reason and Act".
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

# 4. The Agent (Putting it all together)
# We create the agent, which combines the LLM, the prompt, and the tools.
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True lets us see the agent's thoughts in our server logs


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "LangChain Resume Agent API is running."}

@app.post("/analyze")
async def analyze_resume(data: ResumeData):
    try:
        # Create the input string for the agent
        input_text = f"Resume:\n{data.resume_text}\n"
        if data.job_description:
            input_text += f"\nJob Description:\n{data.job_description}\n"
        if data.company_name:
            input_text += f"\nCompany to research:\n{data.company_name}\n"

        # Invoke the agent executor with the input
        response = await agent_executor.ainvoke({"input": input_text})
        
        # The agent's final answer is in the 'output' key
        analysis_data = json.loads(response['output'])

        return {"analysis": analysis_data}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")