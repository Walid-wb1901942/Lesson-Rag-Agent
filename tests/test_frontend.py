from fastapi.testclient import TestClient

from app.main import app


def main():
    client = TestClient(app)
    response = client.get("/")
    print(response.status_code)
    print("Teacher Script Studio" in response.text)


if __name__ == "__main__":
    main()
