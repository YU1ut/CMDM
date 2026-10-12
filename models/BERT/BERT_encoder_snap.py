import os

import torch
import torch.nn as nn


def load_bert(model_path, cond_len=None):
    bert = BERT(model_path, cond_len)
    bert.eval()
    bert.text_model.training = False
    for p in bert.parameters():
        p.requires_grad = False
    return bert


def load_snap_t5(
    model_path: str = "google/t5-v1_1-base",
    cond_len: int = 120,
    local_files_only: bool = False,
):
    t5_encoder = T5TextEncoder(
        model_path=model_path,
        cond_len=cond_len,
        local_files_only=local_files_only,
    )
    t5_encoder.eval()
    t5_encoder.text_model.training = False
    for p in t5_encoder.parameters():
        p.requires_grad = False
    return t5_encoder


class T5TextEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        cond_len: int,
        local_files_only: bool,
    ):
        super().__init__()

        from transformers import AutoTokenizer, T5EncoderModel, logging

        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_max_length = 120 if cond_len is None else int(cond_len)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            legacy=False,
        )
        self.text_model = T5EncoderModel.from_pretrained(
            model_path,
            local_files_only=local_files_only,
        )

    @torch.no_grad()
    def forward(self, texts):
        encoded_inputs = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        device = next(self.text_model.parameters()).device
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)
        output = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]
        return output.detach(), attention_mask.to(dtype=torch.bool)


class BERT(nn.Module):
    def __init__(self, modelpath: str, cond_len):
        super().__init__()

        from transformers import AutoModel, AutoTokenizer, logging

        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.cond_len = cond_len
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        # Text model
        self.text_model = AutoModel.from_pretrained(modelpath)

    def forward(self, texts):
        if self.cond_len is not None:
            encoded_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                max_length=self.cond_len,
                truncation=True,
            )
            output = self.text_model(
                **encoded_inputs.to(self.text_model.device)
            ).last_hidden_state
            mask = encoded_inputs.attention_mask.to(dtype=bool)
            # output = output * mask.unsqueeze(-1)
            return output, mask
        else:
            encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
            output = self.text_model(
                **encoded_inputs.to(self.text_model.device)
            ).last_hidden_state
            mask = encoded_inputs.attention_mask.to(dtype=bool)
            return output, mask
