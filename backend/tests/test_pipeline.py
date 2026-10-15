from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.pipeline import SYSTEM_PROMPT, RagPipeline

REFUSAL = "The information is not available in the current documents."


class FakeIndexer:
    def search(self, question, top_k=4):
        return [
            {"doc_id": "doc_1", "source_url": "https://example.org/1", "category": "Admissions",
             "text": "Applications close on 31 March.", "score": 0.82},
        ][:top_k]


class FakeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Resp", (), {"content": "Applications close on 31 March."})()


def _pipeline():
    p = RagPipeline.__new__(RagPipeline)
    p.indexer = FakeIndexer()
    p.llm = FakeLLM()
    return p


def test_system_prompt_keeps_the_grounding_contract():
    assert "ONLY the provided context" in SYSTEM_PROMPT
    assert "If the answer is not present" in SYSTEM_PROMPT
    # The web UI styles answers by matching this exact sentence.
    assert REFUSAL in SYSTEM_PROMPT


def test_answer_grounds_the_llm_in_retrieved_context():
    p = _pipeline()
    out = p.answer("When do applications close?")

    system, human = p.llm.messages
    assert isinstance(system, SystemMessage) and system.content == SYSTEM_PROMPT
    assert isinstance(human, HumanMessage)
    assert "[doc_id=doc_1 | category=Admissions | source=https://example.org/1]" in human.content
    assert "Applications close on 31 March." in human.content
    assert "Question: When do applications close?" in human.content

    assert out["answer"] == "Applications close on 31 March."
    assert out["used_chunks"] == 1
    assert out["trace"] == [{"doc_id": "doc_1", "source_url": "https://example.org/1", "score": 0.82}]
