import numpy as np
import torch
import transformers
from torch import nn

from .positional_encoding import PositionalEncoding


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModel.from_pretrained(model_name)

        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]


class MotionEncoder(nn.Module):
    def __init__(
        self,
        image_embedding_dim,
        mode: str = "t2m",
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.parts_name = ["Root", "R_Leg", "L_Leg", "Backbone", "R_Arm", "L_Arm"]
        if mode == "t2m":
            parts_input_dim = {
                "Root": 7,
                "R_Leg": 50,
                "L_Leg": 50,
                "Backbone": 60,
                "R_Arm": 60,
                "L_Arm": 60,
            }
        else:
            parts_input_dim = {
                "Root": 7,
                "R_Leg": 62,
                "L_Leg": 62,
                "Backbone": 48,
                "R_Arm": 48,
                "L_Arm": 48,
            }

        for name in self.parts_name:
            raw_dim = parts_input_dim[name]
            latent_dim = image_embedding_dim

            skel_embedding = nn.Linear(raw_dim, image_embedding_dim)

            emb_token = nn.Parameter(torch.randn(latent_dim))

            sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

            seq_trans_encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation,
            )

            seqTransEncoder = nn.TransformerEncoder(
                seq_trans_encoder_layer, num_layers=num_layers
            )

            setattr(self, f"skel_embedding_{name}", skel_embedding)
            setattr(self, f"emb_token_{name}", emb_token)
            setattr(self, f"sequence_pos_encoding_{name}", sequence_pos_encoding)
            setattr(self, f"seqTransEncoder_{name}", seqTransEncoder)

        self.target_token_idx = 0

    def extarct_feature(self, x, name):

        bs, nframes, nfeats = x.shape

        skel_embedding = getattr(self, f"skel_embedding_{name}")
        # Embed each human poses into latent vectors
        x = skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        emb_token = getattr(self, f"emb_token_{name}")

        emb_token = torch.tile(emb_token, (bs,)).reshape(bs, -1)

        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)

        sequence_pos_encoding = getattr(self, f"sequence_pos_encoding_{name}")
        seqTransEncoder = getattr(self, f"seqTransEncoder_{name}")

        # add positional encoding
        xseq = sequence_pos_encoding(xseq)
        final = seqTransEncoder(xseq)

        return final[self.target_token_idx + 1 :]

    def forward(self, parts):
        assert isinstance(parts, list)
        assert len(parts) == len(self.parts_name)

        embedding_parts = []
        for i, name in enumerate(self.parts_name):
            feature = self.extarct_feature(parts[i], name)
            embedding_parts.append(feature.unsqueeze(0))

        embedding_parts = torch.concatenate(embedding_parts, dim=0)
        return embedding_parts.mean(dim=0)


class ClipModel(nn.Module):
    def __init__(
        self,
        text_encoder_alias="distilbert-base-uncased",
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 512,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.5,
        logit: float = 0.07,
        mode: str = "t2m",
    ) -> None:
        super().__init__()

        motion_encoder = MotionEncoder(
            image_embedding_dim=motion_embedding_dims,
            num_layers=4,
            num_heads=4,
            mode=mode,
        )

        text_encoder = TextEncoder(
            model_name=text_encoder_alias, trainable=text_encoder_trainable
        )

        self.motion_encoder = motion_encoder
        self.text_encoder = text_encoder

        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode_motion(self, motion, calc_mean=True):
        motion_features = self.motion_encoder(motion)
        motion_embeddings = self.motion_projection(motion_features)
        if calc_mean:
            return motion_embeddings.mean(dim=0)
        else:
            return motion_embeddings

    def encode_text(self, text):
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )

        text_embeddings = self.text_projection(text_features)

        return text_embeddings

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        motion_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + motion_loss) / 2.0

    def forward(self, motion, text, return_loss=False):
        motion_embeds = self.encode_motion(motion)
        text_embeds = self.encode_text(text)

        # normalized features
        motion_embeds = motion_embeds / motion_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale
        logits_per_motion = logits_per_text.T

        if return_loss:
            return self.clip_loss(logits_per_text)
        else:
            return motion_embeds, text_embeds
