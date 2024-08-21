from __future__ import annotations

import bentoml

@bentoml.service(
    resources={"cpu": "8"}
)
class Ner:
    def __init__(self) -> None:
        import torch
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained("../models/gliner_medium-v2.1", load_onnx=True, load_tokenizer=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

    @bentoml.api()
    def extract(self, text: str, labels: list[str]) -> dict:
        output_dict = {}
        entities = self.model.predict_entities(text, labels, threshold=0.4)
        for entity in entities:
            output_dict[entity["text"]] = entity["label"]
        
        return output_dict
