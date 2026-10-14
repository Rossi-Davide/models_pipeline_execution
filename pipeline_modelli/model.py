from __future__ import annotations

import torch
import torch.nn.functional as F

from .backends import ImageInput, TextInput, available_backends, get_backend


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Model:
    """Universal interface over different vision/vision-language models.

    >>> m = Model("clip")                       # or Model("dinov3")
    >>> m = Model("clip", checkpoint="openai/clip-vit-large-patch14")
    >>> img_emb = m.encode_image(pil_image)     # (N, D)
    >>> txt_emb = m.encode_text(["a cat"])      # CLIP only
    """

    #backend is the underlying model used for operations
    def __init__(
        self,
        name: str,
        checkpoint: str | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        normalize: bool = True,
    ):
        self.name = name
        self.device = device or _auto_device()
        self.normalize = normalize
        self.backend = get_backend(name)(checkpoint, self.device, dtype)

    @staticmethod
    def available() -> list[str]:
        return available_backends()

    @property
    def checkpoint(self) -> str:
        return self.backend.checkpoint

    def _post(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return F.normalize(x, dim=-1) if self.normalize else x

    def encode_image(self, images: ImageInput) -> torch.Tensor:
        return self._post(self.backend.encode_image(images))

    def encode_text(self, texts: TextInput) -> torch.Tensor:
        return self._post(self.backend.encode_text(texts))

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, checkpoint={self.checkpoint!r}, device={self.device!r})"
