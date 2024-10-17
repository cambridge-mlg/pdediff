import math

import torch
import torch.nn as nn
from torch import Size, Tensor
import torch.nn.functional as F


class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi / 2 * 1e3 ** torch.linspace(0, 1, 64))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)



def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  """ from https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layers.py#L515 """

  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb