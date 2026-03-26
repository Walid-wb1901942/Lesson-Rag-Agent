import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

from app.services.ollama_client import generate_text


RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "metadata.json"
TEMPLATE_PATH = RAW_DIR / "metadata_template.json"
LLM_TEMPLATE_PATH = RAW_DIR / "metadata_llm_template.json"
MAX_SAMPLE_CHARS = 4000
MAX_SAMPLE_PAGES = 3


def load_manifest(path: Path) -> dict:
    """Load the metadata manifest JSON from the given path."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_pdfs(folder: Path) -> list[Path]:
    """List all PDF files in a folder, sorted alphabetically."""
    return sorted(path for path in folder.glob("*.pdf") if path.is_file())


def infer_subject(name: str) -> str | None:
    """Infer the academic subject from a PDF filename."""
    lower = name.lower()
    if any(kw in lower for kw in ("math", "integral", "algebra", "calculus", "geometry", "quadratic", "trigonometry")):
        return "mathematics"
    if any(kw in lower for kw in ("english", "literature", "poetry", "grammar")):
        return "literature"
    if any(kw in lower for kw in ("science", "biology", "chemistry", "physics")):
        return "science"
    if "health" in lower:
        return "health"
    return None


def infer_source_type(name: str) -> str:
    """Infer the document source type from the filename."""
    lower = name.lower()
    if "lesson plan" in lower or lower.startswith("term "):
        return "lesson_plan"
    if "activity bank" in lower:
        return "activity_bank"
    if "rubric" in lower:
        return "rubric"
    if "book" in lower:
        return "textbook"
    if "-cg" in lower or "_cg" in lower:
        return "curriculum_guide"
    return "pdf"


def infer_grade_level(name: str) -> str | list[str] | None:
    """Infer the grade level from patterns in the filename."""
    lower = name.lower()

    grade_match = re.search(r"grade[\s_-]*(\d+)", lower)
    if grade_match:
        return grade_match.group(1)

    class_match = re.search(r"class[\s_-]*(\d+)", lower)
    if class_match:
        return class_match.group(1)

    ordinal_match = re.search(r"\b(\d+)(st|nd|rd|th)\b", lower)
    if ordinal_match:
        return ordinal_match.group(1)

    if "english_book" in lower or "english book" in lower:
        return ["9", "10", "11", "12"]

    return None


def infer_topic(name: str) -> str | None:
    """Extract a topic string from the filename after removing noise words."""
    stem = Path(name).stem
    cleaned = stem.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    lower = cleaned.lower()
    for noise in (
        "grade",
        "class",
        "book",
        "lesson plan",
        "activity bank",
        "cg",
        "revised",
        "final",
        "main content",
        "term 1",
    ):
        lower = lower.replace(noise, " ")

    topic = re.sub(r"\s+", " ", lower).strip()
    return topic or None


def build_suggested_metadata(pdf_name: str) -> dict:
    """Build heuristic metadata suggestions for a PDF based on its filename."""
    return {
        "title": Path(pdf_name).stem,
        "subject": infer_subject(pdf_name),
        "grade_level": infer_grade_level(pdf_name),
        "curriculum": "general",
        "topic": infer_topic(pdf_name),
        "source_type": infer_source_type(pdf_name),
        "language": "en",
    }


def build_missing_template(pdf_paths: list[Path], manifest: dict) -> dict:
    """Build a metadata template for PDFs not yet in the manifest."""
    missing: dict[str, dict] = {}
    for pdf_path in pdf_paths:
        if pdf_path.name not in manifest:
            missing[pdf_path.name] = build_suggested_metadata(pdf_path.name)
    return missing


def print_report(pdf_paths: list[Path], manifest: dict) -> None:
    """Print a summary of PDF files and their metadata coverage."""
    missing = [pdf_path.name for pdf_path in pdf_paths if pdf_path.name not in manifest]
    print(f"PDFs found: {len(pdf_paths)}")
    print(f"Metadata entries found: {len(manifest)}")
    print(f"Missing metadata entries: {len(missing)}")
    if missing:
        print("\nMissing PDFs:")
        for name in missing:
            print(f"- {name}")


def write_template(path: Path, template: dict) -> None:
    """Write a metadata template dict to a JSON file."""
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metadata template to {path}")


def extract_pdf_sample(pdf_path: Path, max_pages: int = MAX_SAMPLE_PAGES, max_chars: int = MAX_SAMPLE_CHARS) -> str:
    """Extract text from the first few pages of a PDF for metadata inference."""
    pdf = fitz.open(pdf_path)
    pieces: list[str] = []
    try:
        for page_index in range(min(len(pdf), max_pages)):
            text = pdf[page_index].get_text("text").strip()
            if text:
                pieces.append(f"[Page {page_index + 1}]\n{text}")
            joined = "\n\n".join(pieces)
            if len(joined) >= max_chars:
                return joined[:max_chars]
        return "\n\n".join(pieces)[:max_chars]
    finally:
        pdf.close()


def extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from LLM response text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_llm_metadata_prompt(pdf_name: str, sample_text: str, heuristic_guess: dict) -> str:
    """Build a prompt asking the LLM to infer metadata from a PDF sample."""
    return f"""
You are helping create metadata for educational PDF documents.

Read the PDF sample and infer the best metadata you can.

Return ONLY valid JSON with exactly these keys:
- title
- subject
- grade_level
- curriculum
- topic
- source_type
- language

Rules:
- `subject` should be a short value like "mathematics", "science", "literature", "health", or "general".
- `grade_level` can be a string like "6", a list like ["9", "10", "11", "12"], or null.
- `curriculum` should usually be "general" unless the sample clearly indicates something else.
- `source_type` should be one of: "textbook", "lesson_plan", "activity_bank", "rubric", "curriculum_guide", "notes", or "pdf".
- `language` should be an ISO-like short code such as "en".
- If uncertain, prefer a reasonable best guess rather than leaving every field null.

Filename:
{pdf_name}

Heuristic guess:
{json.dumps(heuristic_guess, ensure_ascii=False)}

PDF sample:
\"\"\"
{sample_text}
\"\"\"
"""


def suggest_metadata_with_llm(pdf_path: Path) -> dict:
    """Use the LLM to infer metadata from a PDF's first pages."""
    heuristic_guess = build_suggested_metadata(pdf_path.name)
    sample_text = extract_pdf_sample(pdf_path)
    if not sample_text:
        return heuristic_guess

    prompt = build_llm_metadata_prompt(pdf_path.name, sample_text, heuristic_guess)
    raw_response = generate_text(prompt)
    parsed = extract_json_object(raw_response)

    if not parsed:
        return heuristic_guess

    normalized = {
        "title": parsed.get("title") or heuristic_guess["title"],
        "subject": parsed.get("subject") or heuristic_guess["subject"],
        "grade_level": parsed.get("grade_level", heuristic_guess["grade_level"]),
        "curriculum": parsed.get("curriculum") or heuristic_guess["curriculum"],
        "topic": parsed.get("topic") or heuristic_guess["topic"],
        "source_type": parsed.get("source_type") or heuristic_guess["source_type"],
        "language": parsed.get("language") or heuristic_guess["language"],
    }
    return normalized


def build_llm_missing_template(pdf_paths: list[Path], manifest: dict, limit: int | None = None) -> dict:
    """Generate LLM-assisted metadata suggestions for PDFs missing from the manifest."""
    missing_paths = [pdf_path for pdf_path in pdf_paths if pdf_path.name not in manifest]
    if limit is not None:
        missing_paths = missing_paths[:limit]

    template: dict[str, dict] = {}
    for pdf_path in missing_paths:
        print(f"LLM metadata suggestion for {pdf_path.name}...")
        template[pdf_path.name] = suggest_metadata_with_llm(pdf_path)
    return template


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the metadata tools."""
    parser = argparse.ArgumentParser(description="Audit and scaffold metadata for raw PDFs.")
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write heuristic metadata suggestions for missing PDFs to data/raw/metadata_template.json",
    )
    parser.add_argument(
        "--write-llm-template",
        action="store_true",
        help="Write LLM-assisted metadata suggestions for missing PDFs to data/raw/metadata_llm_template.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for how many missing PDFs to process when writing the LLM template.",
    )
    return parser


def main() -> None:
    """Entry point for the metadata audit CLI."""
    args = build_parser().parse_args()
    manifest = load_manifest(MANIFEST_PATH)
    pdf_paths = list_pdfs(RAW_DIR)

    print_report(pdf_paths, manifest)

    if args.write_template:
        template = build_missing_template(pdf_paths, manifest)
        write_template(TEMPLATE_PATH, template)

    if args.write_llm_template:
        template = build_llm_missing_template(pdf_paths, manifest, limit=args.limit)
        write_template(LLM_TEMPLATE_PATH, template)


if __name__ == "__main__":
    main()
