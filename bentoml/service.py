import os
import bentoml
import onnxruntime
from src.word_process.inputs_outputs_process import DataProcess
import torch


@bentoml.service(resources={"cpu": "8"})
class Ner:
    def __init__(self):
        model_path = "../models/gliner_medium-v2.1"
        onnx_model_path = os.path.join(model_path, "model.onnx")

        self.data_process = DataProcess(model_path)
        self.ort_sess = onnxruntime.InferenceSession(onnx_model_path)

    @bentoml.api
    def extract(self, text: str, labels: list[str]) -> dict:
        inputs, raw_batch = self.data_process.prepare_model_inputs([text], labels)
        output_dict = {}
        outputs = self.ort_sess.run(
            None,
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy(),
                "words_mask": inputs["words_mask"].numpy(),
                "text_lengths": inputs["text_lengths"].numpy(),
                "span_idx": inputs["span_idx"].numpy(),
                "span_mask": inputs["span_mask"].numpy(),
            },
        )[0]
        outputs = torch.from_numpy(outputs)
        outputs = self.data_process.decode(raw_batch["tokens"], raw_batch["id_to_classes"], torch.tensor(outputs))[0]

        texts = raw_batch["tokens"][0]
        for output in outputs:
            start, end = output[:2]
            entity = output[2]
            text_entity = " ".join(texts[start : end + 1])
            output_dict[text_entity] = entity

        return output_dict
