# pipeline_modelli

One `Model` class to drive different models (CLIP, DINOv3).

```python
from PIL import Image
from pipeline_modelli import Model

clip = Model("clip")
img = clip.encode_image(Image.open("a.jpg"))
txt = clip.encode_text(["a photo of a dog"])
print(Model.similarity(img, txt))

dino = Model("dinov3", checkpoint="facebook/dinov3-vitb16-pretrain-lvd1689m")
feats = dino.encode_image([Image.open("a.jpg")])
```

DINOv3 checkpoints on Hugging Face are gated: accept the license and run `huggingface-cli login`.

## Adding a model

Subclass `Backend` in `pipeline_modelli/backends.py`, implement `load` / `encode_image` / `encode_text`, and decorate with `@register("name")`.
