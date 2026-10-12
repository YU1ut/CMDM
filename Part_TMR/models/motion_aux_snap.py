"""Motion half of a SNAP Part-TMR checkpoint for auxiliary VAE training."""

from torch import nn

from .clip_snap import MotionEncoder, ProjectionHead


class MotionAuxModel(nn.Module):
    def __init__(
        self,
        motion_embedding_dims=512,
        projection_dims=256,
        dropout=0.0,
        mode="snapmogen",
    ):
        super().__init__()
        self.motion_encoder = MotionEncoder(
            image_embedding_dim=motion_embedding_dims,
            num_layers=4,
            num_heads=4,
            mode=mode,
        )
        self.motion_projection = ProjectionHead(
            embedding_dim=motion_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )

    def load_part_tmr_state_dict(self, state_dict):
        motion_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("motion_encoder.", "motion_projection."))
        }
        self.load_state_dict(motion_state, strict=True)

    def encode_motion(self, motion, calc_mean=True):
        motion_features = self.motion_encoder(motion)
        motion_embeddings = self.motion_projection(motion_features)
        return motion_embeddings.mean(dim=0) if calc_mean else motion_embeddings
