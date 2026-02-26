"""Example 2: Multi-model streaming pipeline.

Demonstrates Nerva's core value proposition — orchestrating multiple
models in a DAG with dynamic batching and streaming output.

Pipeline: Tokenizer → LLM → Detokenizer
           (CPU)     (GPU)     (CPU)

This is the target API for the full MVP.
"""

import nerva
from nerva import Model, batch, model, serve, stream, trace


# --- Step 1: Define model implementations ---


class ToyTokenizer(Model):
    """Converts text string to token IDs."""

    def load(self) -> None:
        # In real usage, load a tokenizer (e.g., sentencepiece, tiktoken)
        self.vocab = {chr(i): i for i in range(256)}

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        text = inputs["text"]
        assert isinstance(text, str)
        token_ids = [self.vocab.get(c, 0) for c in text]
        return {"token_ids": token_ids}


class ToyLLM(Model):
    """Toy autoregressive generator — echoes input tokens with offset.

    In real usage, this would be backed by vLLM's AsyncLLMEngine.
    """

    def load(self) -> None:
        self.max_new_tokens = 32

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        token_ids = inputs["token_ids"]
        assert isinstance(token_ids, list)
        # Toy: generate by shifting each token by +1
        generated = [(t + 1) % 256 for t in token_ids[: self.max_new_tokens]]
        return {"generated_ids": generated}

    # Streaming variant — yield tokens one by one
    async def infer_stream(self, inputs: dict[str, object]):  # type: ignore[override]
        token_ids = inputs["token_ids"]
        assert isinstance(token_ids, list)
        for t in token_ids[: self.max_new_tokens]:
            yield {"token_id": (t + 1) % 256}


class ToyDetokenizer(Model):
    """Converts token IDs back to text."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        token_ids = inputs["generated_ids"]
        assert isinstance(token_ids, list)
        text = "".join(chr(t) for t in token_ids)
        return {"text": text}


# --- Step 2: Declare models ---

tokenizer = model("tokenizer", ToyTokenizer, backend="pytorch", device="cpu")
llm = model("llm", ToyLLM, backend="pytorch", device="cpu")  # use vllm + cuda in prod
detokenizer = model("detokenizer", ToyDetokenizer, backend="pytorch", device="cpu")


# --- Step 3: Define pipeline function ---


def text_generation(text: object) -> object:
    tokens = tokenizer(text)
    output = llm(tokens)
    return detokenizer(output)


# --- Step 4: Apply transforms ---

graph = trace(text_generation)
graph = batch(graph, targets=["llm"], max_size=32, max_delay_ms=10)
graph = stream(graph)
app = serve(graph, route="/rpc/text_generation")


# --- Expected usage ---
#
# Start server:
#   nerva run examples/02_multi_model_pipeline.py
#
# Unary call:
#   result = client.call("text_generation", {"text": "Hello world"})
#   print(result["text"])
#
# Streaming call:
#   for chunk in client.stream("text_generation", {"text": "Hello world"}):
#       print(chunk["token_id"], end="", flush=True)
