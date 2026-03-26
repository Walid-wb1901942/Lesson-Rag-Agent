from app.services.lesson_generator import generate_lesson_from_query


def main():
    result = generate_lesson_from_query("Write a crypto trading bot with risk controls.")

    print("=" * 100)
    print("GENERATION MODE")
    print("=" * 100)
    print(result["generation_mode"])
    print(result["source_notice"])

    print("\n" + "=" * 100)
    print("RESPONSE")
    print("=" * 100)
    print(result["lesson_text"])


if __name__ == "__main__":
    main()
