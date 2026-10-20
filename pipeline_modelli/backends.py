"""Backend implementations wrapped by :class:`pipeline_modelli.Model`.

    To add a new model create a class that inherits from Backend with the model checkpoint
    and specific loading mechanisms. Register decorator automatically adds it to registry

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Type, Union

import torch
from PIL import Image

ImageInput = Union[Image.Image, Sequence[Image.Image]]
TextInput = Union[str, Sequence[str]]

_REGISTRY: Dict[str, Type["Backend"]] = {}


def register(name: str) -> Callable[[Type["Backend"]], Type["Backend"]]:
    def deco(cls: Type["Backend"]) -> Type["Backend"]:
        _REGISTRY[name] = cls
        return cls

    return deco


def available_backends() -> List[str]:
    return sorted(_REGISTRY)


def get_backend(name: str) -> Type["Backend"]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {available_backends()}") from None


def _as_list(x):
    return [x] if isinstance(x, (str, Image.Image)) else list(x)


class Backend(ABC):
    default_checkpoint: str = ""

    def __init__(self, checkpoint: str | None, device: str, dtype: torch.dtype | None):
        self.checkpoint = checkpoint or self.default_checkpoint
        self.device = device
        self.dtype = dtype
        self.load()

    @abstractmethod
    def load(self) -> None: ...

    def encode_image(self, images: ImageInput) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not support image encoding")

    def encode_text(self, texts: TextInput) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not support text encoding")

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            v = v.to(self.device)
            if self.dtype is not None and v.is_floating_point():
                v = v.to(self.dtype)
            out[k] = v
        return out


def _as_tensor(out) -> torch.Tensor:
    """transformers>=5 returns an output object (projected features in pooler_output), older versions a tensor."""
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    return out[0]


@register("clip")
class CLIPBackend(Backend):
    default_checkpoint = "openai/clip-vit-base-patch32"

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(self.checkpoint)
        device = torch.device(self.device)
        self.model = CLIPModel.from_pretrained(self.checkpoint, torch_dtype=self.dtype).to(device).eval()

    @torch.inference_mode()
    def encode_image(self, images: ImageInput) -> torch.Tensor:
        inputs = self._to_device(self.processor(images=_as_list(images), return_tensors = "pt"))
        return _as_tensor(self.model.get_image_features(**inputs))

    @torch.inference_mode()
    def encode_text(self, texts: TextInput) -> torch.Tensor:
        inputs = self._to_device(
            self.processor(text=_as_list(texts), return_tensors="pt", padding=True, truncation=True)
        )
        return _as_tensor(self.model.get_text_features(**inputs))


@register("dinov3")
class DINOv3Backend(Backend):
    default_checkpoint = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModel.from_pretrained(self.checkpoint, torch_dtype=self.dtype).to(self.device).eval()

    @torch.inference_mode()
    def encode_image(self, images: ImageInput) -> torch.Tensor:
        inputs = self._to_device(self.processor(images=_as_list(images), return_tensors="pt"))
        return self.model(**inputs).pooler_output  # CLS token embedding
