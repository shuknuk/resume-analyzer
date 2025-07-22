# ✨ Intelligent Resume Analyzer - Backend

This repository contains the backend for the Intelligent Resume Analyzer. This is not just a simple API; it's the "brain" of the application—a true **AI Agent** built with Python, FastAPI, and LangChain.

[**➡️ View the Live API Endpoint**](https://resume-analyzer-czxr.onrender.com)

## Core Logic: The ReAct Agent

This backend implements the **ReAct (Reason and Act)** framework to create an autonomous agent. Unlike a simple API call, the agent can:

1.  **Reason:** Analyze a user's request and determine if it needs more information to provide a high-quality answer.
2.  **Act:** Autonomously use tools to gather new information. Currently, it uses the **Tavily Search API** to browse the live internet for research on companies.
3.  **Synthesize:** Combine its internal knowledge, the user's documents, and the new information from its tools to generate a deeply contextual and strategic analysis.

## Tech Stack

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
* **Language:** [Python](https://www.python.org/)
* **AI Agent Framework:** [LangChain](https://www.langchain.com/)
* **LLM:** [Google Gemini 2.5 Flash](https://ai.google.dev/models/gemini)
* **Agent Tools:** [Tavily Search API](https://tavily.com/)
* **Containerization:** [Docker](https://www.docker.com/)
* **Deployment:** [Render](https://render.com/)

## API Endpoint

The primary endpoint for the application is:

* `POST /analyze`
    * **Request Body:**
        ```json
        {
          "resume_text": "...",
          "job_description": "...",
          "company_name": "..."
        }
        ```
    * **Response Body:**
        ```json
        {
          "analysis": {
            "score": 85,
            "scoreRationale": "...",
            "strengths": ["...", "..."],
            "improvements": [
              {
                "suggestion": "...",
                "explanation": "...",
                "example": "..."
              }
            ]
          },
          "log": "Thought: ...\nAction: ...\nObservation: ..."
        }
        ```

## Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shuknuk/resume-analyzer.git
    cd resume-analyzer
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add your API keys:
    ```
    GOOGLE_API_KEY="your-google-api-key"
    TAVILY_API_KEY="your-tavily-api-key"
    ```

5.  **Run the development server:**
    ```bash
    uvicorn main:app --reload
    ```

The API will be available at `http://localhost:8000`.

### Frontend Repository

The user interface for this project is a Next.js application located in a separate repository.

[**➡️ View Frontend Repo**](https://github.com/shuknuk/resume-analyzer-frontend)

> ### View Case Study
>
> You can read a full case study detailing the architecture, challenges, and key learnings from this project on my personal portfolio.
>
> [**➡️ Read the Case Study at kinshuk-goel.vercel.app**](https://kinshuk-goel.vercel.app/)