"""Example 3: Parallel DAG with conditional routing.

Demonstrates nerva.parallel() and nerva.cond() control flow primitives.

Pipeline:
  ┌─ ImageEncoder ─┐
  │                 ├─→ FusionModel → Classifier
  └─ TextEncoder  ─┘

With conditional routing:
  if media_type == "image": use ImageEncoder only
  else: use both encoders (multimodal fusion)
"""

import nerva
from nerva import Model, model, serve, trace

# --- Model implementations ---


class ToyImageEncoder(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        # Toy: just return a fixed-size "feature vector"
        data = inputs["image_bytes"]
        assert isinstance(data, bytes)
        features = [float(b) / 255.0 for b in data[:64]]
        return {"features": features}


class ToyTextEncoder(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        text = inputs["text"]
        assert isinstance(text, str)
        features = [float(ord(c)) / 255.0 for c in text[:64]]
        return {"features": features}


class ToyFusionModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        img_feat = inputs["img_features"]
        txt_feat = inputs["txt_features"]
        assert isinstance(img_feat, list) and isinstance(txt_feat, list)
        # Toy fusion: element-wise average
        min_len = min(len(img_feat), len(txt_feat))
        fused = [
            (a + b) / 2.0
            for a, b in zip(img_feat[:min_len], txt_feat[:min_len], strict=False)
        ]
        return {"fused_features": fused}


class ToyClassifier(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, object]) -> dict[str, object]:
        features = inputs.get("fused_features") or inputs.get("features")
        assert isinstance(features, list)
        score = sum(features) / max(len(features), 1)
        label = "positive" if score > 0.5 else "negative"
        return {"label": label, "score": score}


# --- Model declarations ---

image_encoder = model("image_encoder", ToyImageEncoder, backend="pytorch", device="cpu")
text_encoder = model("text_encoder", ToyTextEncoder, backend="pytorch", device="cpu")
fusion = model("fusion", ToyFusionModel, backend="pytorch", device="cpu")
classifier = model("classifier", ToyClassifier, backend="pytorch", device="cpu")


# --- Pipeline with parallel execution ---


def multimodal_classify(request: object) -> object:
    assert isinstance(request, dict)

    # Parallel: run both encoders concurrently
    img_feat, txt_feat = nerva.parallel(
        lambda: image_encoder({"image_bytes": request["image"]}),
        lambda: text_encoder({"text": request["text"]}),
    )

    fused = fusion({"img_features": img_feat["features"], "txt_features": txt_feat["features"]})
    return classifier(fused)


# --- Pipeline with conditional routing ---


def adaptive_classify(request: object) -> object:
    assert isinstance(request, dict)

    features = nerva.cond(
        request.get("media_type") == "image_only",
        # True branch: image encoder only
        lambda: image_encoder({"image_bytes": request["image"]}),
        # False branch: full multimodal fusion
        lambda: multimodal_classify(request),
    )

    return nerva.cond(
        "fused_features" in features or "features" in features,
        lambda: classifier(features),
        lambda: features,  # already classified
    )


# --- Apply transforms and serve ---

graph = trace(multimodal_classify)
app = serve(graph, route="/rpc/multimodal_classify")


# --- Expected usage ---
#
# Multimodal request:
#   result = client.call("multimodal_classify", {
#       "image": b"\x00\x01\x02...",
#       "text": "a photo of a cat",
#   })
#   print(result["label"])  # "positive" or "negative"
