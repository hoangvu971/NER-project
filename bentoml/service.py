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




# from __future__ import annotations

# import bentoml.service
# import numpy
# import torch
# import bentoml
# from bentoml.io import JSON
# from src.word_process.inputs_outputs_process import DataProcess

# model_path = "../models/gliner_medium-v2.1"
# data_process = DataProcess(model_path)

# runner = bentoml.onnx.get("ner:latest").to_runner()
# svc = bentoml.Service("onnx_super_resolution", runners=[runner])

# @bentoml.api(input=JSON(), output=JSON())
# def extract(data: dict) -> dict:
#     text = data.get("text")
#     labels = data.get("labels")
#     inputs, raw_batch = data_process.prepare_model_inputs([text], labels)
#     outputs = runner.run.run(inputs['input_ids'].numpy(),
#                                 inputs['attention_mask'].numpy(),
#                                 inputs['words_mask'].numpy(),
#                                 inputs['text_lengths'].numpy(),
#                                 inputs['span_idx'].numpy(),
#                                 inputs['span_mask'].numpy(),
#                                 )[0]
    
#     outputs = data_process.decode(raw_batch["tokens"], raw_batch["id_to_classes"], torch.tensor(outputs))

#     outputs_dict = {}
#     texts = raw_batch['tokens'][0]
#     for output in outputs:
#         start, end = output[:2]
#         entity = output[2]
#         outputs_dict[texts[start:end+1]] = entity

#     return outputs_dict
