import json

import httpx
import pytest

from app.config import Settings
from app.generate import (REFUSAL, Generation, GenerationError, OllamaProvider, OpenAIProvider,
                          build_prompt, extract_citations, generate_answer, is_refusal, make_provider)
from app.ingest import Chunk
from app.store import SearchResult

RESULTS = [
    SearchResult(Chunk("Paris is the capital of France.", "geo.pdf", 3, 7), score=0.8),
    SearchResult(Chunk("Berlin is the capital of Germany.", "geo.pdf", 5, 12), score=0.6),
    SearchResult(Chunk("The Seine flows through Paris.", "rivers.pdf", 1, 0), score=0.4),
]


class FakeProvider:
    """Satisfies the LLMProvider protocol without inheriting from anything."""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def generate(self, system: str, user: str) -> Generation:
        self.calls.append((system, user))
        return Generation(self.reply, "fake-model", prompt_tokens=100, completion_tokens=10)


# --- prompt ---

def test_prompt_numbers_passages_with_source_and_page_then_asks_question():
    system, user = build_prompt("What is the capital of France?", RESULTS)
    assert REFUSAL in system
    assert "[1] (source: geo.pdf, page 3)\nParis is the capital of France." in user
    assert "[3] (source: rivers.pdf, page 1)" in user
    assert user.index("[3]") < user.index("Question: What is the capital of France?")


# --- citations and refusals ---

def test_citations_map_numbers_to_chunks_in_order_of_mention():
    cited = extract_citations("Paris [3] is the capital [1][3], see also [1, 2].", RESULTS)
    assert [c.chunk.chunk_index for c in cited] == [0, 7, 12]  # passages 3, 1, 2


def test_citations_ignore_numbers_that_are_not_passages():
    assert extract_citations("As shown in [0] and [9], and in 2020.", RESULTS) == []


@pytest.mark.parametrize("text, expected", [
    (REFUSAL, True),
    ("I don’t know based on the provided documents.", True),  # curly apostrophe
    ("Paris is the capital [1].", False),
])
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected


def test_generate_answer_attaches_citations():
    answer = generate_answer("capital of France?", RESULTS, FakeProvider("It is Paris [1]."))
    assert answer.answered is True
    assert [c.chunk.page for c in answer.citations] == [3]
    assert answer.retrieved == RESULTS
    assert answer.generation.prompt_tokens == 100


def test_generate_answer_refusal_has_no_citations():
    answer = generate_answer("capital of Peru?", RESULTS, FakeProvider(REFUSAL))
    assert answer.answered is False
    assert answer.citations == []


# --- OpenAI provider (real SDK, fake HTTP transport) ---

def openai_with(handler) -> OpenAIProvider:
    return OpenAIProvider("sk-test", "gpt-4o-mini", timeout=5,
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_openai_provider_sends_messages_and_parses_usage():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini-2024-07-18",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": " Paris [1]. "}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 4, "total_tokens": 124},
        })

    result = openai_with(handler).generate("SYSTEM", "USER")
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["temperature"] == 0
    assert result == Generation("Paris [1].", "gpt-4o-mini-2024-07-18", 120, 4)


def test_openai_auth_error_becomes_generation_error():
    provider = openai_with(lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(GenerationError, match="AuthenticationError"):
        provider.generate("s", "u")


# --- Ollama provider (fake HTTP transport) ---

def ollama_with(handler) -> OllamaProvider:
    return OllamaProvider("http://ollama.test", "llama3.2", timeout=5,
                          transport=httpx.MockTransport(handler))


def test_ollama_provider_sends_non_streaming_chat_and_parses_token_counts():
    sent = {}

    def handler(request):
        sent["path"] = request.url.path
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "llama3.2", "message": {"role": "assistant", "content": "Paris [1]."},
            "prompt_eval_count": 95, "eval_count": 6, "done": True,
        })

    result = ollama_with(handler).generate("SYSTEM", "USER")
    assert sent["path"] == "/api/chat"
    assert sent["stream"] is False and sent["options"]["temperature"] == 0
    assert result == Generation("Paris [1].", "llama3.2", 95, 6)


def test_ollama_missing_model_error_is_readable():
    provider = ollama_with(lambda request: httpx.Response(404, json={"error": "model 'llama3.2' not found"}))
    with pytest.raises(GenerationError, match="404.*not found"):
        provider.generate("s", "u")


def test_ollama_connection_refused_becomes_generation_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(GenerationError, match="ConnectError"):
        ollama_with(handler).generate("s", "u")


# --- provider selection ---

def test_make_provider_requires_openai_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_provider(Settings(llm_provider="openai", openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_provider(Settings(llm_provider="openai", openai_api_key=""))


def test_make_provider_selects_by_env_setting():
    assert isinstance(make_provider(Settings(llm_provider="ollama")), OllamaProvider)
    assert isinstance(make_provider(Settings(llm_provider="openai", openai_api_key="sk-x")), OpenAIProvider)
