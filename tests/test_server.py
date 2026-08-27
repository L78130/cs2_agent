# tests/test_server.py
import io
from fastapi.testclient import TestClient
from demo_coach.web import server


def test_matches_empty():
    server.CONTEXTS.clear()
    c = TestClient(server.app)
    assert c.get("/api/matches").json() == []


def test_chat_unknown_demo():
    server.CONTEXTS.clear()
    c = TestClient(server.app)
    r = c.post("/api/matches/nope/chat", json={"message": "hi", "history": []})
    assert r.status_code == 404


def test_upload_rejects_non_dem():
    c = TestClient(server.app)
    r = c.post("/api/upload", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
    assert r.status_code == 400
