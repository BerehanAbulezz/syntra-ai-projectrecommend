import json
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq, APIConnectionError, APIStatusError, RateLimitError

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Student Project Suggestion Service",
    description="Generates tailored project ideas for students using the Groq LLM API.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class ProjectRequest(BaseModel):
    track: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="The student's learning track (e.g. 'Full-Stack Web Development').",
        examples=["Backend Development"],
    )
    technologies: list[str] = Field(
        ...,
        min_length=1,
        description="List of technologies the student is working with.",
        examples=[["Python", "FastAPI", "PostgreSQL"]],
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Project Advisor for software-engineering students.

Rules you MUST follow without exception:
1. Respond ONLY with a single, valid JSON object — no markdown fences, no prose, no extra keys.
2. Maintain absolute technical accuracy: never mix incompatible layers
   (e.g. React is a frontend framework — do NOT suggest it as a backend tool;
    Django/FastAPI are backend — do NOT suggest them as frontend frameworks).
3. Every project must realistically use the technologies the student provided.
4. Difficulty must be exactly one of: "Intermediate" or "Advanced".
5. Return between 3 and 5 projects inside the "projects" array."""

USER_PROMPT_TEMPLATE = """Generate project suggestions for a student with the following profile:

Track       : {track}
Technologies: {technologies}

Return a JSON object that matches this exact schema (no extra fields):

{{
  "projects": [
    {{
      "project_title": "string — concise, descriptive title",
      "difficulty": "Intermediate | Advanced",
      "overview": "string — 2-3 sentence description of what the project does",
      "core_features": ["string", "string", "..."],
      "tech_stack_usage": "string — how the given technologies are specifically used",
      "implementation_steps": ["string", "string", "..."]
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Groq client (initialised once; reads GROQ_API_KEY from the environment)
# ---------------------------------------------------------------------------

def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: GROQ_API_KEY environment variable is not set.",
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@app.post(
    "/api/projects/generate",
    summary="Generate project suggestions",
    response_description="A JSON object containing a list of tailored project ideas.",
)
async def generate_projects(request: ProjectRequest) -> dict:
    """
    Accepts a student's **track** and **technologies**, calls the Groq LLM,
    and returns a structured list of project suggestions instantly.

    No data is stored — this endpoint is completely stateless.
    """
    client = _get_groq_client()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        track=request.track,
        technologies=", ".join(request.technologies),
    )

    # -----------------------------------------------------------------------
    # Call Groq
    # -----------------------------------------------------------------------
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
    except RateLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=f"Groq rate limit reached. Please try again shortly. ({exc})",
        ) from exc
    except APIConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the Groq API. Check your network. ({exc})",
        ) from exc
    except APIStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API returned an error: {exc.status_code} — {exc.message}",
        ) from exc

    # -----------------------------------------------------------------------
    # Parse the LLM's JSON response
    # -----------------------------------------------------------------------
    raw_content: str = completion.choices[0].message.content or ""

    try:
        result: dict = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"The model returned a response that is not valid JSON. "
                f"Parse error: {exc}. Raw content: {raw_content[:300]}"
            ),
        ) from exc

    if "projects" not in result or not isinstance(result["projects"], list):
        raise HTTPException(
            status_code=500,
            detail=(
                "The model returned a JSON object with an unexpected structure. "
                f"Keys received: {list(result.keys())}"
            ),
        )

    return result


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------

@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "ok"}