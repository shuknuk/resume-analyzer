# main.py
import os
import json
import re # Import re for regex in parsing
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
# Note: Removed trailing spaces from origins
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
# These lines fetch from .env and set them, which is standard practice.
# Ensure your .env file has GOOGLE_API_KEY=your_key_here and TAVILY_API_KEY=your_key_here
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


# --- Pydantic Models ---
class ResumeData(BaseModel):
    resume_text: str
    job_description: str | None = None
    company_name: str | None = None


# --- LangChain Agent Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3) # Confirmed model name
tools = [TavilySearchResults(max_results=3)]

# --- Improved Prompt ---
# Made the format instructions stricter and clearer.
prompt = PromptTemplate.from_template("""
You are an expert career coach and professional resume strategist. Your goal is to provide a detailed, objective, and constructive analysis of a resume.

You have access to the following tools:
{tools}

You MUST use the following format EXACTLY, with each component on a separate line:

Question: the input question you must answer
Thought: you should always think about what to do. NEVER concatenate 'Thought:' with your actual thought (e.g., NEVER do 'Thought:Thought: ...'). Just write 'Thought:' followed by your reasoning on the next line.
Action: the action to take, should be one of [{tool_names}]. This MUST be on its own line immediately after 'Thought:'.
Action Input: the input to the action. This MUST be on its own line immediately after 'Action:'.
Observation: the result of the action. This will be provided to you.
... (this Thought/Action/Action Input/Observation cycle can repeat N times)
Thought: I now know the final answer. This MUST be on its own line.
Final Answer: [Your complete analysis here as a JSON object]

IMPORTANT: Your Final Answer must be a valid JSON object with this exact structure:
{{
  "score": 85,
  "scoreRationale": "Strong technical skills and experience match the role requirements well",
  "strengths": ["Strong technical expertise in required technologies", "Proven leadership and mentoring experience"],
  "improvements": [
    {{
      "suggestion": "Quantify Achievements",
      "explanation": "Adding specific metrics makes your impact more compelling to employers",
      "example": "Instead of: 'Improved system performance' Try: 'Improved system performance by 40% through database optimization'"
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
    handle_parsing_errors=True, # Keep this for robustness
    return_intermediate_steps=True,
    max_iterations=5,
    early_stopping_method="generate"
)

def _parse_json_from_string(text: str) -> dict:
    """Extract and parse JSON from agent response with fallback handling."""
    print(f"[_parse_json_from_string] Received text for parsing (first 500 chars): {text[:500]}...")
    try:
        # --- Robust JSON Extraction using Regex ---
        # This pattern looks for the outermost braces of a JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
             print(f"[_parse_json_from_string] No JSON object pattern found in response.")
             raise ValueError("No JSON object pattern found")

        json_str = json_match.group(0)
        print(f"[_parse_json_from_string] Extracted JSON string: {json_str[:200]}...")
        parsed_json = json.loads(json_str)

        # Validate required fields
        required_fields = ["score", "scoreRationale", "strengths", "improvements"]
        for field in required_fields:
            if field not in parsed_json:
                raise ValueError(f"Missing required field in parsed JSON: {field}")

        print(f"[_parse_json_from_string] Successfully parsed JSON.")
        return parsed_json

    except (json.JSONDecodeError, ValueError, re.error) as e:
        print(f"[_parse_json_from_string] Error parsing JSON from agent response: {e}")
        # Log the raw text for debugging if it's not massive
        if len(text) < 2000: # Avoid printing extremely long strings
             print(f"[_parse_json_from_string] Raw response text: {text}")
        else:
             print(f"[_parse_json_from_string] Raw response text (truncated): {text[:2000]}...")

        # Return a fallback response
        return {
            "score": 70,
            "scoreRationale": "Analysis completed with parsing issues in the response format.",
            "strengths": ["Resume shows relevant experience", "Contains key technical skills"],
            "improvements": [
                {
                    "suggestion": "Response Processing",
                    "explanation": "The analysis system encountered formatting issues in generating the detailed response.",
                    "example": "Please try submitting your request again for a more detailed analysis."
                }
            ]
        }

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "LangChain Resume Agent API is running."}

async def _fallback_analysis(resume_text: str, job_description: str = None, company_name: str = None) -> dict:
    """Fallback analysis using direct LLM call without agent framework."""
    print("[_fallback_analysis] Starting fallback analysis...")
    try:
        analysis_prompt = f"""
You are an expert career coach. Analyze this resume and provide feedback in JSON format.

Resume:
{resume_text}

{f"Job Description: {job_description}" if job_description else ""}
{f"Company: {company_name}" if company_name else ""}

Respond with ONLY a valid JSON object with this exact structure:
{{
  "score": 85,
  "scoreRationale": "Brief explanation of the score",
  "strengths": ["strength 1", "strength 2"],
  "improvements": [
    {{
      "suggestion": "Improvement area",
      "explanation": "Why this matters",
      "example": "Specific example of how to improve"
    }}
  ]
}}
Do not include any other text, explanations, or markdown. Only the JSON object.
"""
        print(f"[_fallback_analysis] Sending prompt to LLM...")
        response = await llm.ainvoke(analysis_prompt)
        print(f"[_fallback_analysis] Received response from LLM (first 500 chars): {response.content[:500]}...")
        result = _parse_json_from_string(response.content)
        print(f"[_fallback_analysis] Fallback analysis successful.")
        return result
    except Exception as e:
        print(f"[_fallback_analysis] Fallback analysis failed: {e}")
        # Return a basic fallback if the fallback itself fails
        return {
            "score": 75,
            "scoreRationale": "Basic analysis completed via fallback mechanism.",
            "strengths": ["Resume submitted successfully", "Contains relevant information"],
            "improvements": [
                {
                    "suggestion": "System Processing",
                    "explanation": "The detailed analysis system is currently experiencing issues.",
                    "example": "Please try again later for a more detailed analysis."
                }
            ]
        }

@app.post("/analyze")
async def analyze_resume(data: ResumeData):
    print(f"[/analyze] Received request for analysis.")
    # Build the input text dynamically
    input_text = f"Analyze this resume:\n---{data.resume_text}\n---"
    if data.job_description and data.job_description.strip():
        input_text += f"\nTailor the analysis for this specific job description:\n---{data.job_description}\n---"
    if data.company_name and data.company_name.strip():
        input_text += f"\nAlso, provide extra insights by researching this company:\n---{data.company_name}\n---"
    print(f"[/analyze] Constructed input text (first 500 chars): {input_text[:500]}...")

    # --- Attempt Agent Analysis ---
    try:
        print("[/analyze] Attempting analysis with LangChain Agent...")
        response = await agent_executor.ainvoke({"input": input_text})

        # Process intermediate steps for logging
        log_string = ""
        for step in response.get('intermediate_steps', []):
            try:
                action, observation = step
                # Try different attribute names for robustness
                thought_log = getattr(action, 'log', '')
                if thought_log:
                     # Extract the last 'Thought:' line if multiple or messy
                    thought_lines = [line for line in thought_log.split('\n') if line.strip().startswith('Thought:')]
                    if thought_lines:
                         log_string += f"{thought_lines[-1].strip()}\n"
                    else:
                         log_string += f"Thought: {thought_log.strip()}\n" # Fallback

                action_name = getattr(action, 'tool', getattr(action, 'name', 'Unknown Action'))
                log_string += f"Action: {action_name}\n"

                action_input = getattr(action, 'tool_input', '')
                log_string += f"Action Input: {action_input}\n"

                log_string += f"Observation: {observation}\n\n"
            except Exception as log_error:
                print(f"[/analyze] Error processing intermediate log step: {log_error}")
                log_string += f"Raw Step (Error processing): {step}\n\n"

        # Try to parse the agent's final output
        agent_output = response.get('output', '')
        print(f"[/analyze] Agent finished. Raw output (first 500 chars): {agent_output[:500]}...")
        analysis_data = _parse_json_from_string(agent_output)
        print("[/analyze] Agent analysis parsed successfully.")
        return {"analysis": analysis_data, "log": log_string}

    except Exception as agent_error:
        print(f"[/analyze] Agent execution failed: {agent_error}")
        # --- Fallback to Direct LLM Analysis ---
        try:
            print("[/analyze] Attempting fallback analysis...")
            analysis_data = await _fallback_analysis(
                data.resume_text,
                data.job_description,
                data.company_name
            )
            print("[/analyze] Fallback analysis completed successfully.")
            # Return fallback result with a log indicating fallback was used
            return {
                "analysis": analysis_data,
                "log": f"Agent failed, used fallback analysis. Agent Error: {str(agent_error)}"
            }
        except Exception as fallback_error:
             print(f"[/analyze] Fallback analysis also failed: {fallback_error}")
             # If fallback also fails, raise HTTP 500
             error_detail = f"Analysis failed: Agent error: {str(agent_error)}, Fallback error: {str(fallback_error)}"
             print(f"[/analyze] Raising HTTP 500 with detail: {error_detail}")
             raise HTTPException(status_code=500, detail=error_detail)

    # This point should ideally not be reached due to the structure above,
    # but as a final safety net, catch any unexpected errors in the outer scope:
    # except Exception as unexpected_error: 
    #     print(f"[/analyze] Unexpected error occurred: {unexpected_error}")
    #     raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(unexpected_error)}")

