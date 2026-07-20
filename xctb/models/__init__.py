"""Model components. Everything here imports torch; import lazily if you are in
a torch-free environment (the split and deferral code does not need any of it).
"""

from xctb.models.model import TriageModel, build_model

__all__ = ["TriageModel", "build_model"]
