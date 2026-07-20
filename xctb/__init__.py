"""xctb: cross-cohort, uncertainty-aware TB chest X-ray triage.

The pieces that make this project different from a normal classifier live in
two modules that have no torch dependency, so they run and test anywhere:

    xctb.data.splits    leave-one-cohort-out split construction + leakage checks
    xctb.eval.deferral  risk-coverage curves and the generalization-gap
                        recovery metric

Everything that touches a GPU (backbones, training, MC-dropout inference) is
isolated under xctb.models and xctb.engine and only imports torch when used.
"""

__version__ = "0.1.0"
