from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.responses import FileResponse

from app.schemas import (
    LessonChatRequest,
    LessonRequest,
    LessonResponse,
    QuizRequest,
    QuizResponse,
    TTSRequest,
)
from app.services.chatbot import LessonScriptChatbot
from app.services.pipeline import ScriptPipeline
from app.services.quiz_pipeline import QuizPipeline

app = FastAPI(
    title="Lesson RAG Agent API",
    description="API for generating educational lessons with RAG.",
    version="0.1.0",
)

pipeline = ScriptPipeline()
chatbot = LessonScriptChatbot(pipeline=pipeline)
quiz_pipeline = QuizPipeline()
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index() -> FileResponse:
    """Serve the built-in browser chat UI."""
    return FileResponse(STATIC_DIR / "index.html")



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
    result = pipeline.run(
        user_prompt=request.user_prompt,
        subject=request.subject,
        grade_level=request.grade_level,
        topic=request.topic,
        retrieval_limit=request.retrieval_limit,
        retrieval_mode=request.retrieval_mode,
        retrieval_method=request.retrieval_method,
    )
    return LessonResponse(**result)


@app.post("/lessons/generate", response_model=LessonResponse)
def generate_lesson(request: LessonRequest) -> LessonResponse:
    """Alias for /agent/run."""
    return run_agent(request)


_script_jobs: dict[str, dict] = {}


def _run_script(job_id: str, request: LessonChatRequest):
    """Background task that runs chatbot generation and stores the result."""
    try:
        result = chatbot.chat(
            message=request.message,
            current_script=request.current_script,
            original_request=request.original_request,
            subject=request.subject,
            grade_level=request.grade_level,
            topic=request.topic,
            retrieval_limit=request.retrieval_limit,
            retrieval_mode=request.retrieval_mode,
            retrieval_method=request.retrieval_method,
        )
        _script_jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        import traceback
        print(f"SCRIPT ERROR for job {job_id}:\n{traceback.format_exc()}")
        _script_jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/chat/script")
def chat_script(request: LessonChatRequest, background_tasks: BackgroundTasks) -> dict:
    """Start script generation in the background. Returns a job ID to poll."""
    job_id = str(uuid4())[:8]
    _script_jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(_run_script, job_id, request)
    return {"job_id": job_id, "status": "processing"}


@app.get("/chat/status/{job_id}")
def chat_status(job_id: str) -> dict:
    """Check the status of a script generation job."""
    job = _script_jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {"job_id": job_id, **job}


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


# ---- Quiz Generation ----
_quiz_jobs: dict[str, dict] = {}


@app.post("/quiz/generate", response_model=QuizResponse)
def generate_quiz_sync(request: QuizRequest) -> QuizResponse:
    """Synchronously generate a RAG-grounded quiz. Use for quick requests."""
    result = quiz_pipeline.run(
        content=request.content,
        subject=request.subject,
        grade_level=request.grade_level,
        num_questions=request.num_questions,
        difficulty=request.difficulty,
        question_types=request.question_types,
        retrieval_limit=request.retrieval_limit,
        retrieval_mode=request.retrieval_mode,
        retrieval_method=request.retrieval_method,
    )
    return QuizResponse(**result)


def _run_quiz(job_id: str, request: QuizRequest) -> None:
    """Background task that runs quiz generation and stores the result."""
    try:
        result = quiz_pipeline.run(
            content=request.content,
            subject=request.subject,
            grade_level=request.grade_level,
            num_questions=request.num_questions,
            difficulty=request.difficulty,
            question_types=request.question_types,
            retrieval_limit=request.retrieval_limit,
            retrieval_mode=request.retrieval_mode,
            retrieval_method=request.retrieval_method,
        )
        _quiz_jobs[job_id] = {"status": "done", "result": result}
    except Exception as e:
        import traceback
        print(f"QUIZ ERROR for job {job_id}:\n{traceback.format_exc()}")
        _quiz_jobs[job_id] = {"status": "error", "error": str(e)}


@app.post("/quiz/start")
def start_quiz(request: QuizRequest, background_tasks: BackgroundTasks) -> dict:
    """Start quiz generation in the background. Returns a job ID to poll."""
    job_id = str(uuid4())
    _quiz_jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(_run_quiz, job_id, request)
    return {"job_id": job_id, "status": "processing"}


@app.get("/quiz/status/{job_id}")
def quiz_status(job_id: str) -> dict:
    """Check the status of a quiz generation job."""
    job = _quiz_jobs.get(job_id)
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
