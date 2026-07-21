# Survey: calibrated uncertainty methods for small models

Chris's Phase-1 task. Goal: pick uncertainty/calibration methods that (a) are
cheap enough to stay edge-deployable and (b) actually feed a deferral policy
that can be evaluated with `xctb.eval.deferral`'s gap-recovery metric. Three of
the four candidate methods are already implemented in this repo; this doc
explains each, why it fits (or doesn't) a compressed TB-Net-scale model, and
flags the one gap.

## 1. Temperature scaling (Guo et al., 2017, "On Calibration of Modern Neural
Networks")

- **What it does**: fit one scalar T post-hoc on validation logits to minimize
  NLL, then divide logits by T before softmax. Doesn't change which class wins
  (argmax is invariant to T > 0), only how confident the softmax looks.
- **Cost**: one extra scalar, fit once, offline. Zero added inference cost,
  zero added parameters in the deployed model.
- **Where it lives here**: `xctb/calibration.py` — `fit_temperature` (golden-
  section search on validation NLL), `apply_temperature`, and
  `expected_calibration_error` for measuring the fix. Already used in the
  README's protocol: fit T on the validation split of the *seen* cohorts,
  apply it to the held-out cohort at test time.
- **Fit for this project**: essential baseline. It's the cheapest possible
  calibration fix and the natural first thing to check before reaching for
  anything model-side. Its limitation is that it's a single global scalar — it
  can't fix a model that's confidently wrong in different *directions* on
  different cohorts, which is exactly the cross-site failure mode we're
  studying. That's the motivation for pairing it with an uncertainty signal
  (below) rather than relying on it alone.

## 2. MC-dropout (Gal & Ghahramani, 2016, "Dropout as a Bayesian
Approximation")

- **What it does**: leave dropout active at inference, run T stochastic
  forward passes, treat the spread of the T predictions as an uncertainty
  estimate (approximates Bayesian model averaging under a variational
  interpretation of dropout).
- **Cost**: no extra parameters and no retraining — only requires the backbone
  already has dropout layers. Inference cost multiplies by T (this repo
  defaults to T=20 in `infer.py::predict`), which is the real tradeoff for an
  edge/offline device: T forward passes instead of one.
- **Where it lives here**: `xctb/engine/infer.py::predict(method="mc_dropout")`
  — `_enable_dropout` flips dropout back to train-mode behavior at eval time,
  then uncertainty is `std` of the positive-class probability across samples.
- **Fit for this project**: good default for the deferral experiments because
  it needs no architecture change and reuses whatever dropout the backbone
  already has. T=20 is a meaningful latency multiplier on a phone-class
  device, so it's worth treating T as a tunable knob (report AURC vs. T) rather
  than fixing it at 20 without justification.

## 3. Deep ensembles (Lakshminarayanan et al., 2017, "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles")

- **What it does**: train M independently-initialized copies of the model,
  ensemble their predictions; disagreement across members is the uncertainty
  signal. Widely reported as the strongest-calibrated of the classical methods,
  usually beating MC-dropout on both accuracy and calibration.
- **Cost**: M full models' worth of parameters and M forward passes — the
  worst-case cost of the group for a model whose whole selling point is being
  small enough for edge deployment. M x 4.24M params (TB-Net scale) or M x
  ~230K (Pasa-et-al scale) may still be fine on-device if M is small (3-5) and
  storage, not compute, is the binding constraint; if latency is binding,
  MC-dropout's T passes over one model is usually cheaper than M passes over M
  models unless T > M.
- **Where it lives here**: `xctb/engine/infer.py::ensemble_predict` — same
  output contract as `predict`, uncertainty is `std` across members' positive-
  class probability.
- **Fit for this project**: worth running as the "best-case calibration" upper
  bound to compare MC-dropout against, but probably not the headline method
  given the edge-deployable framing — report it as a ceiling, recommend
  MC-dropout or temperature scaling as the deployable choice unless the
  ensemble's gap-recovery number is meaningfully better.

## 4. Evidential deep learning (Sensoy, Kaplan & Kandemir, 2018, "Evidential
Deep Learning to Quantify Classification Uncertainty", NeurIPS)

- **What it does**: replace the softmax output with parameters of a Dirichlet
  distribution over class probabilities, trained with an evidence-based loss
  (subjective logic framing) instead of cross-entropy. Uncertainty falls out
  of the Dirichlet directly (its concentration/"total evidence") — no sampling
  and no extra forward passes needed at inference.
- **Cost**: single forward pass at inference (cheapest of the three uncertainty
  methods here), but it requires **changing the loss function and the output
  head, and retraining from scratch** — it isn't a drop-in wrapper around a
  trained classifier the way MC-dropout and ensembling are.
- **Where it lives here**: **not implemented.** `xctb/models/model.py` outputs
  standard 2-class logits and `xctb/engine/train.py` trains with what's
  presumably a standard cross-entropy-style objective; there's no Dirichlet
  head or evidential loss anywhere in `xctb/losses/`.
- **Fit for this project**: the most attractive method on paper for an
  edge-deployment story (one pass, no ensembling) but the highest-effort to
  add correctly, and evidential losses are known to need careful annealing of
  the KL/regularization term to train stably — a real risk for a project on a
  paper deadline. Recommendation: **flag as future work / a stretch goal**
  rather than a Phase-3 requirement. If time allows after MC-dropout, temp
  scaling, and the ensemble baseline are all running through `run_loco.py`,
  this is the natural next thing to add to `xctb/losses/` and
  `xctb/models/model.py` (new head) — but don't block the paper's core LOCO +
  deferral result on it.

## Recommendation for Phase 3

Ship MC-dropout + temperature scaling as the primary pipeline (already wired
end to end: `infer.py` -> `calibration.py` -> `eval/deferral.py`), run deep
ensembles as a comparison ceiling, and write up evidential deep learning in the
paper's future-work / limitations section rather than implementing it now.
This keeps every reported number reproducible with what's already in the repo
and matches the "torch-free core stays fast to iterate on" philosophy in
`ONBOARDING.md`.

## Sources

- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." ICML.
- Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning." ICML.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and
  Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.
- Sensoy, M., Kaplan, L., & Kandemir, M. (2018). "Evidential Deep Learning to
  Quantify Classification Uncertainty." NeurIPS.
  https://papers.nips.cc/paper/7580-evidential-deep-learning-to-quantify-classification-uncertainty
