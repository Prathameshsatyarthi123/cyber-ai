import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from context import get_analysis_prompt, enhance_summary

load_dotenv()

app = FastAPI(title="Cybersecurity Analyzer API")

cors_origins = [
    "http://localhost:3000",
    "http://frontend:3000",
]

if os.getenv("ENVIRONMENT") == "production":
    cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    code: str


class SecurityIssue(BaseModel):
    title: str = Field(description="Brief title of the security vulnerability")
    description: str = Field(description="Detailed description of the security issue and its potential impact")
    code: str = Field(description="The specific vulnerable code snippet that demonstrates the issue")
    fix: str = Field(description="Recommended code fix or mitigation strategy")
    cvss_score: float = Field(description="CVSS score from 0.0 to 10.0 representing severity")
    severity: str = Field(description="Severity level: critical, high, medium, or low")


class SecurityReport(BaseModel):
    summary: str = Field(description="Executive summary of the security analysis")
    issues: List[SecurityIssue] = Field(description="List of identified security vulnerabilities")


def validate_request(request: AnalyzeRequest) -> None:
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="No code provided for analysis")


def check_api_keys() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="Groq API key not configured")


def get_groq_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1"
    )


async def run_security_analysis(code: str) -> SecurityReport:
    client = get_groq_client()

    system_prompt = """You are an expert security researcher. Analyze the provided code for security vulnerabilities.
You must respond with valid JSON only, matching this exact structure:
{
  "summary": "executive summary string",
  "issues": [
    {
      "title": "vulnerability title",
      "description": "detailed description",
      "code": "vulnerable code snippet",
      "fix": "recommended fix",
      "cvss_score": 7.5,
      "severity": "high"
    }
  ]
}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_analysis_prompt(code)}
        ],
        response_format={"type": "json_object"}
    )

    import json
    data = json.loads(response.choices[0].message.content)
    return SecurityReport(**data)


def format_analysis_response(code: str, report: SecurityReport) -> SecurityReport:
    enhanced_summary = enhance_summary(len(code), report.summary)
    return SecurityReport(summary=enhanced_summary, issues=report.issues)


@app.post("/api/analyze", response_model=SecurityReport)
async def analyze_code(request: AnalyzeRequest) -> SecurityReport:
    validate_request(request)
    check_api_keys()

    try:
        report = await run_security_analysis(request.code)
        return format_analysis_response(request.code, report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health():
    return {"message": "Cybersecurity Analyzer API"}


@app.get("/network-test")
async def network_test():
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("https://semgrep.dev/api/v1/")
            return {
                "semgrep_api_reachable": True,
                "status_code": response.status_code,
                "response_size": len(response.content)
            }
    except Exception as e:
        return {"semgrep_api_reachable": False, "error": str(e)}


if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
