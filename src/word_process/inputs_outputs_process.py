import json
from pathlib import Path

from .collator import DataCollator
from .processor import SpanProcessor, SpanBiEncoderProcessor, TokenProcessor, TokenBiEncoderProcessor
from .tokenizer import WordsSplitter
from .config import GLiNERConfig

from ..decoding.decoder import SpanDecoder, TokenDecoder

from transformers import AutoTokenizer


class DataProcess:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._set_up_data_processor()

    def _set_up_data_processor(self):
        config_file = Path(self.model_dir) / "gliner_config.json"
        config_ = json.load(open(config_file))
        config = GLiNERConfig(**config_)

        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        words_splitter = WordsSplitter(config.words_splitter_type)

        if config.span_mode == "token_level":
            if config.labels_encoder is not None:
                labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder)
                self.data_processor = TokenBiEncoderProcessor(config, tokenizer, words_splitter, labels_tokenizer)
            else:
                self.data_processor = TokenProcessor(config, tokenizer, words_splitter)
            self.decoder = TokenDecoder(config)
        else:
            if config.labels_encoder is not None:
                labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder)
                self.data_processor = SpanBiEncoderProcessor(config, tokenizer, words_splitter, labels_tokenizer)
            else:
                self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
            self.decoder = SpanDecoder(config)

        add_tokens = ["[FLERT]", config.ent_token, config.sep_token]
        if config.class_token_index == -1 or config.vocab_size == -1:
            self.data_processor.transformer_tokenizer.add_tokens(add_tokens)

    def prepare_model_inputs(self, texts: list[str], labels: list[str], prepare_entities: bool = True):
        """
        Prepare inputs for the model.

        Args:
            texts (str): The input text or texts to process.
            labels (str): The corresponding labels for the input texts.
        """
        all_tokens = []
        all_start_token_idx_to_text_idx = []
        all_end_token_idx_to_text_idx = []
        # preserving the order of labels
        labels = list(dict.fromkeys(labels))

        class_to_ids = {k: v for v, k in enumerate(labels, start=1)}
        id_to_classes = {k: v for v, k in class_to_ids.items()}

        for text in texts:
            tokens = []
            start_token_idx_to_text_idx = []
            end_token_idx_to_text_idx = []
            for token, start, end in self.data_processor.words_splitter(text):
                tokens.append(token)
                start_token_idx_to_text_idx.append(start)
                end_token_idx_to_text_idx.append(end)
            all_tokens.append(tokens)
            all_start_token_idx_to_text_idx.append(start_token_idx_to_text_idx)
            all_end_token_idx_to_text_idx.append(end_token_idx_to_text_idx)

        input_x = [{"tokenized_text": tk, "ner": None} for tk in all_tokens]
        raw_batch = self.data_processor.collate_raw_batch(
            input_x, labels, class_to_ids=class_to_ids, id_to_classes=id_to_classes
        )
        raw_batch["all_start_token_idx_to_text_idx"] = all_start_token_idx_to_text_idx
        raw_batch["all_end_token_idx_to_text_idx"] = all_end_token_idx_to_text_idx

        model_input = self.data_processor.collate_fn(raw_batch, prepare_labels=False, prepare_entities=prepare_entities)
        model_input.update(
            {
                "span_idx": raw_batch["span_idx"] if "span_idx" in raw_batch else None,
                "span_mask": raw_batch["span_mask"] if "span_mask" in raw_batch else None,
                "text_lengths": raw_batch["seq_length"],
            }
        )

        return model_input, raw_batch

    def decode(self, tokens, id_to_classes, model_output, flat_ner=True, threshold=0.5, multi_label=False):
        return self.decoder.decode(tokens, id_to_classes, model_output, flat_ner, threshold, multi_label)
