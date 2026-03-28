from app.services.pipeline import ScriptPipeline


def main():
    agent = ScriptPipeline()
    result = agent.run(
        user_prompt="Create a 40-minute Grade 12 English lesson on Shakespeare with a class activity and short exit ticket.",
        subject="literature",
        grade_level="12",
        topic="",
        retrieval_limit=5,
    )

    print("=" * 100)
    print("GENERATION MODE")
    print("=" * 100)
    print(result["generation_mode"])
    print(result["source_notice"])
    print("Evaluation:", result["evaluation"])

    print("=" * 100)
    print("GENERATED LESSON SCRIPT")
    print("=" * 100)
    print(result["lesson_text"])

    print("\n" + "=" * 100)
    print("RETRIEVED CHUNKS USED")
    print("=" * 100)

    for i, chunk in enumerate(result["retrieved_chunks"], start=1):
        print(f"\nSource {i}")
        print("Score:", chunk["score"])
        print("Metadata:", chunk["metadata"])
        print("Text preview:", chunk["text"][:300])


if __name__ == "__main__":
    main()
