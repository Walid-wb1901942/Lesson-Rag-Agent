from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.responses import FileResponse

from app.schemas import LessonChatRequest, LessonChatResponse, LessonRequest, LessonResponse, TTSRequest
from app.services.agent import LessonPlanningAgent
from app.services.chatbot import LessonScriptChatbot

app = FastAPI(
    title="Lesson RAG Agent API",
    description="API for generating educational lessons with RAG and agent orchestration.",
    version="0.1.0",
)

agent = LessonPlanningAgent()
chatbot = LessonScriptChatbot(agent=agent)
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index() -> FileResponse:
    """Serve the built-in browser chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/old")
def index_old() -> FileResponse:
    """Serve the legacy browser UI."""
    return FileResponse(STATIC_DIR / "index_old.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Return empty response for favicon requests."""
    return Response(status_code=204)


@app.get("/health")
def healthcheck() -> dict:
    """Return API health status."""
    return {"status": "ok"}


@app.post("/agent/run", response_model=LessonResponse)
def run_agent(request: LessonRequest) -> LessonResponse:
    """Generate a lesson script using the agent pipeline."""
    result = agent.run(
        user_prompt=request.user_prompt,
        subject=request.subject,
        grade_level=request.grade_level,
        curriculum=request.curriculum,
        topic=request.topic,
        retrieval_limit=request.retrieval_limit,
        retrieval_mode=request.retrieval_mode,
    )
    return LessonResponse(**result)


@app.post("/lessons/generate", response_model=LessonResponse)
def generate_lesson(request: LessonRequest) -> LessonResponse:
    """Alias for /agent/run."""
    return run_agent(request)


@app.post("/chat/script", response_model=LessonChatResponse)
def chat_script(request: LessonChatRequest) -> LessonChatResponse:
    """Chat-style endpoint for generating and revising lesson scripts."""
    result = chatbot.chat(
        message=request.message,
        current_script=request.current_script,
        original_request=request.original_request,
        subject=request.subject,
        grade_level=request.grade_level,
        curriculum=request.curriculum,
        topic=request.topic,
        retrieval_limit=request.retrieval_limit,
        retrieval_mode=request.retrieval_mode,
    )
    return LessonChatResponse(**result)


# ---- TTS Audio Generation ----
AUDIO_DIR = Path(__file__).parent / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Track generation status: job_id -> {"status": "processing"|"done"|"error", "path": ...}
_tts_jobs: dict[str, dict] = {}


def _run_tts(job_id: str, script: str, engine: str):
    """Background task that generates audio."""
    try:
        from app.services.tts import generate_lesson_audio

        output_path = AUDIO_DIR / f"{job_id}"
        result_path = generate_lesson_audio(script, output_path, engine=engine)
        _tts_jobs[job_id] = {"status": "done", "path": str(result_path)}
    except Exception as e:
        import traceback
        print(f"TTS ERROR for job {job_id}:\n{traceback.format_exc()}")
        _tts_jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/tts/generate")
def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks) -> dict:
    """Start audio generation in the background. Returns a job ID to poll."""
    job_id = str(uuid4())[:8]
    _tts_jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(_run_tts, job_id, request.script, request.engine)
    return {"job_id": job_id, "status": "processing"}


@app.get("/tts/status/{job_id}")
def tts_status(job_id: str) -> dict:
    """Check the status of a TTS generation job."""
    job = _tts_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {"job_id": job_id, **job}


@app.get("/tts/download/{job_id}")
def tts_download(job_id: str) -> FileResponse:
    """Download the generated audio file."""
    job = _tts_jobs.get(job_id)
    if not job or job["status"] != "done":
        return Response(status_code=404, content="Audio not ready")
    file_path = Path(job["path"])
    ext = file_path.suffix  # .mp3 or .wav
    media = "audio/mpeg" if ext == ".mp3" else "audio/wav"
    return FileResponse(
        job["path"],
        media_type=media,
        filename=f"lesson_{job_id}{ext}",
    )
