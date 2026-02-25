"""Example 1: Single model unary inference.

Demonstrates the simplest Nerva usage — one PyTorch model serving
unary (request-response) inference over HTTP.

This is the target API for Phase 0. The example should be runnable
once Phase 0 + Phase 4 are complete.
"""

import torch
import torch.nn as nn

import nerva
from nerva import Model, model, serve, trace


# --- Step 1: Define model implementation ---


class SentimentClassifier(Model):
    """Toy sentiment classifier for demonstration."""

    def load(self) -> None:
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )
        self.net.eval()

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        """Single-request inference.

        Args:
            inputs: {"embedding": Tensor of shape [seq_len, 128]}

        Returns:
            {"scores": Tensor of shape [2], "label": str}
        """
        embedding = inputs["embedding"]
        assert isinstance(embedding, torch.Tensor)
        with torch.inference_mode():
            pooled = embedding.mean(dim=0, keepdim=True)  # [1, 128]
            scores = self.net(pooled).squeeze(0)  # [2]
        label = "positive" if scores[1] > scores[0] else "negative"
        return {"scores": scores, "label": label}


# --- Step 2: Declare model (lazy — not loaded until serve) ---

classifier = model(
    "sentiment",
    SentimentClassifier,
    backend="pytorch",
    device="cpu",
)


# --- Step 3: Define pipeline (trivial — single model) ---


def classify(text_embedding: object) -> object:
    return classifier(text_embedding)


# --- Step 4: Apply transforms and serve ---

graph = trace(classify)
app = serve(graph, route="/rpc/classify")


# --- Expected usage ---
#
# Start server:
#   nerva run examples/01_single_model.py
#
# or programmatically:
#   uvicorn examples.01_single_model:app
#
# Client call (pseudo-code, using future client SDK):
#   import nerva.client
#   client = nerva.client.connect("http://localhost:8000")
#   result = client.call("classify", {"embedding": tensor})
#   print(result["label"])  # "positive" or "negative"
