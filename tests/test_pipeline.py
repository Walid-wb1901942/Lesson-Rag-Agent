from app.services.pipeline import ScriptPipeline


def main():
    pipeline = ScriptPipeline()
    result = pipeline.run(
        user_prompt="Create a 40-minute Grade 6 science lesson script on ecosystems and food chains with a class activity and short exit ticket.",
        subject="science",
        grade_level="6",
        topic="ecosystems",
    )
    print("=" * 100)
    print("PIPELINE MODE")
    print("=" * 100)
    print(result["generation_mode"])
    print(result["source_notice"])

    print("\n" + "=" * 100)
    print("PIPELINE EVALUATION")
    print("=" * 100)
    print(result["evaluation"])

    print("\n" + "=" * 100)
    print("PIPELINE TRACE")
    print("=" * 100)
    for step in result["agent_trace"]:
        print(step["tool"], step)

    print("\n" + "=" * 100)
    print("LESSON SCRIPT")
    print("=" * 100)
    print(result["lesson_text"])


if __name__ == "__main__":
    main()
