from pydantic import BaseModel, Field
from uuid import uuid4

GradeLevel = str | list[str] | None


class DocumentMetadata(BaseModel):
    """Metadata attached to a source PDF document."""

    source_path: str
    file_name: str
    file_type: str
    title: str | None = None
    subject: str | None = None
    grade_level: GradeLevel = None
    topic: str | None = None
    source_type: str | None = None
    language: str = "en"
    page_number: int | None = None


class RawDocument(BaseModel):
    """A full PDF document with merged page text and metadata."""

    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: DocumentMetadata


class ChunkMetadata(DocumentMetadata):
    """Extended metadata for a document chunk, including position info."""

    doc_id: str
    chunk_index: int
    page_range: str | None = None


class DocumentChunk(BaseModel):
    """A text chunk extracted from a document, ready for embedding."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: ChunkMetadata


class LessonRequest(BaseModel):
    """Request body for generating a lesson script."""

    user_prompt: str
    subject: str | None = None
    grade_level: str | None = None
    topic: str | None = None
    retrieval_limit: int = 5
    retrieval_mode: str = "auto"
    retrieval_method: str = "dense"


class Citation(BaseModel):
    """A single source citation extracted from generated lesson text."""

    source_number: int
    title: str | None = None
    pages: str | None = None
    subject: str | None = None
    grade_level: str | None = None
    topic: str | None = None


class LessonResponse(BaseModel):
    """Response body for a generated lesson script with evaluation and trace."""

    user_prompt: str
    generation_mode: str
    source_notice: str
    retrieved_chunks: list[dict]
    prompt_used: str | None = None
    lesson_text: str
    evaluation: dict | None = None
    agent_trace: list[dict]
    citations: list[Citation] = []


class LessonChatRequest(BaseModel):
    """Request body for chatbot-style script generation and revision."""

    message: str
    current_script: str | None = None
    original_request: str | None = None
    subject: str | None = None
    grade_level: str | None = None
    topic: str | None = None
    retrieval_limit: int = 5
    retrieval_mode: str = "auto"
    retrieval_method: str = "dense"


class LessonChatResponse(BaseModel):
    """Response body for chatbot-style interactions including revision results."""

    chat_mode: str
    assistant_message: str
    follow_up_prompt: str | None = None
    user_prompt: str
    generation_mode: str
    source_notice: str
    retrieved_chunks: list[dict]
    prompt_used: str | None = None
    lesson_text: str
    evaluation: dict | None = None
    agent_trace: list[dict]
    citations: list[Citation] = []


class TTSRequest(BaseModel):
    """Request body for text-to-speech audio generation."""

    script: str
    engine: str = "edge"  # "edge" (fast) or "bark" (emotional)
