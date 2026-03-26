from fastapi.testclient import TestClient

from app.main import app


def main():
    client = TestClient(app)

    health_response = client.get("/health")
    print("Health:", health_response.status_code, health_response.json())

    lesson_response = client.post(
        "/agent/run",
        json={
            "user_prompt": "Write a crypto trading bot",
        },
    )
    print("Agent run:", lesson_response.status_code)
    print(lesson_response.json())


if __name__ == "__main__":
    main()
