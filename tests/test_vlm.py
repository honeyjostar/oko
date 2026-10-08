"""Клиент VLM проверяется на поддельном OpenAI-совместимом сервере — без GPU и без сети."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from oko.engines.vlm import VLMEngine, json_schema, parse_answer
from oko.schema import DOC_FIELDS

ANSWER = {"doc_type": "sts_front", "fields": {
    "plate": "a123bc164", "vin": "xta21170ok1234567", "make_model": "КIА RIO", "year": "2019",
    "owner_last": None, "doc_number": "99 21 123456", "unknown_key": "мусор"}}


def _server(mode):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"data": [{"id": "qwen3-vl-8b"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(req)
            if mode == "no_schema" and "response_format" in req:
                return self._send(400, {"error": "response_format is not supported"})
            text = json.dumps(ANSWER, ensure_ascii=False)
            if mode == "fenced":
                text = "<think>смотрю на документ</think>Вот результат:\n```json\n" + text + "\n```"
            if mode == "garbage":
                text = "не могу прочитать"
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": text}}]})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.mark.parametrize("mode", ["schema_ok", "no_schema", "fenced"])
def test_extract_through_openai_api(mode):
    srv, seen = _server(mode)
    try:
        eng = VLMEngine(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="qwen3-vl-8b")
        assert eng.available() == (True, "")
        r = eng.extract(Image.new("RGB", (2400, 1600), "white"))
    finally:
        srv.shutdown()
    assert r.error is None
    assert r.doc_type == "sts_front"
    assert r.fields == {"plate": "А123ВС164", "vin": "XTA211700K1234567", "make_model": "KIA RIO",
                        "year": "2019", "doc_number": "99 21 123456"}
    img_url = seen[-1]["messages"][0]["content"][1]["image_url"]["url"]
    assert img_url.startswith("data:image/jpeg;base64,")
    assert seen[-1]["temperature"] == 0
    if mode == "no_schema":
        assert "response_format" in seen[0] and "response_format" not in seen[-1]


def test_unreadable_answer_is_an_error_not_a_crash():
    srv, _ = _server("garbage")
    try:
        r = VLMEngine(base_url=f"http://127.0.0.1:{srv.server_port}/v1").extract(Image.new("RGB", (100, 100)))
    finally:
        srv.shutdown()
    assert r.fields == {} and r.error


def test_server_down():
    eng = VLMEngine(base_url="http://127.0.0.1:9/v1")
    ok, why = eng.available()
    assert not ok and "LM Studio" in why
    assert eng.extract(Image.new("RGB", (50, 50))).error


def test_parse_answer_variants():
    assert parse_answer('{"a": 1}') == {"a": 1}
    assert parse_answer('текст ```json\n{"a": 2}\n``` текст') == {"a": 2}
    assert parse_answer('<think>{"x": 0}</think> {"a": 3}') == {"a": 3}
    assert parse_answer("нет json") is None


def test_schema_covers_all_fields():
    props = json_schema()["schema"]["properties"]["fields"]["properties"]
    assert set(props) == {k for keys in DOC_FIELDS.values() for k in keys}
