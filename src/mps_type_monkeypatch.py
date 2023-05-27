import torch
from torch import Tensor
import logging

logger = logging.getLogger(__name__)

def monkeypatch_tensor_type_mps():
  # Falcon's modelling_RW.py`RotaryEmbedding#cos_sin has a hardcoded dtype=torch.bfloat16 param, which is no bueno for MPS
  # https://huggingface.co/tiiuae/falcon-7b-instruct/blob/b6efaea5d78e4313145bda4d688675414a76fc22/modelling_RW.py#L71
  # I don't know how to import RotaryEmbedding (it doesn't begin on our PYTHONPATH; Huggingface libraries import it somehow later)
  # but I *do* know how to monkeypatch torch.Tensor.type, so let's gooo
  if torch.backends.mps.is_available():
    logger.warning("Monkey-patching Tensor#type to avoid casting to bfloat16 (and prefer float16), to get around a hardcoded cast which wouldn't work on Mac.")
    orig_type = Tensor.type
    def monkeypatched_type(self: Tensor, *args, **kwargs):
      if args:
        first, *rest = args
        if first is torch.bfloat16:
          return orig_type(self, torch.float16, *rest, **kwargs)
      return orig_type(self, *args, **kwargs)
    Tensor.type = monkeypatched_type