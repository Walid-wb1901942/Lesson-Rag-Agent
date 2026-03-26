from fastapi.testclient import TestClient

from app.main import app


def main():
    client = TestClient(app)

    response = client.post(
        "/chat/script",
        json={
            "message": "Write a crypto trading bot",
        },
    )
    print(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
