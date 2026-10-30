"""WavLM, vendored from https://github.com/nicolvisser/ZeroSyl (MIT).

`WavLM.py` and `modules.py` are copied unmodified from `zerosyl/wavlm/` at commit
1daf45809571d87b9a6ef8c24e90f2bcb85a3b7e; only a provenance header was added. The
licence text is in `LICENSE` next to this file.

Re-exported lazily: importing this module needs `torch`, which most installs of
`hohere` (Part A) do not have.
"""

from __future__ import annotations

__all__ = ["WavLM", "WavLMConfig"]


def __getattr__(name: str):
    """`from hohere.vendor.wavlm import WavLM, WavLMConfig` without an eager torch import."""
    if name in __all__:
        from .WavLM import WavLM, WavLMConfig

        return {"WavLM": WavLM, "WavLMConfig": WavLMConfig}[name]
    raise AttributeError(name)
