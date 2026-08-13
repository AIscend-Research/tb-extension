# Deployment checklist

What a clinic would actually need before using this model on a real patient.
Borrowed in structure from clinical implementation science (and from the
reporting discipline of CONSORT-AI / DECIDE-AI / TRIPOD+AI), narrowed to the
specific failure modes this project studies.

**Current status: nothing below is satisfied. This is a research tool, not a
clinical device, and it is not for diagnosis.** The list is here so the gap is
explicit rather than implied.

---

## A. Before the model is allowed to output anything

- [ ] **Reported on a site that contributed no training data.** A LOCO fold, not
      a random split. Random-split accuracy on pooled data (TB-Net's 99.86%) does
      not predict behaviour at a new clinic and must not be quoted as if it does.
- [ ] **Sensitivity target set by public-health need, not by F1.** A missed TB
      case seeds onward transmission; a false positive costs one confirmatory
      test. Fix the operating point on sensitivity, then report the specificity
      that buys it — `eval.target: sensitivity` in the config.
- [ ] **Calibration measured, not assumed.** ECE/MCE and a reliability diagram on
      the deployment site's own data. An overconfident model makes deferral
      actively dangerous, because the cases it should escalate are exactly the
      ones it will report confidently.
- [ ] **Temperature fitted on site-local validation data**, not inherited from the
      development site. If the fitted temperature lands on a search bound,
      `metrics.json` flags `temperature_at_bound` — stop and investigate.
- [ ] **Prevalence documented.** Every probability output is conditioned on the
      training prevalence. A screening population at 0.5% TB is not the ~50% of
      Montgomery/Shenzhen, and predictive values shift accordingly even with
      identical sensitivity and specificity.

## B. The deferral pathway must exist before deferral is switched on

- [ ] **A named person receives deferred cases**, with a defined response time.
      "Refer to specialist" is not a feature unless there is a specialist.
- [ ] **Deferral rate budgeted against staffing.** Decide the maximum fraction of
      cases the site can absorb, then set the threshold to that coverage — not the
      reverse. `eval.min_coverage` encodes this constraint.
- [ ] **"Retake the photo" and "refer to a specialist" are distinguished.** They
      are different actions with different costs. Degradation-driven uncertainty
      should trigger a retake; genuine diagnostic ambiguity should trigger a
      referral. `eval/conformal.py` separates these as `no_plausible_label` vs.
      `ambiguous`.
- [ ] **A retake actually helps.** If the failure is a cracked lightbox or a
      broken camera, retaking produces another bad photo. Cap retakes and escalate.
- [ ] **Deferral is not silently disabled** when the queue gets long. This is the
      predictable failure mode under workload pressure, and it converts the system
      from safe to dangerous without any code change.

## C. Monitoring, once live

- [ ] **Per-site drift monitoring running.** `eval/sequential_deferral.CUSUMMonitor`
      over the stream of per-image uncertainty, with a defined alarm response.
      Cameras degrade, lightboxes get replaced, staff change.
- [ ] **Input-quality distribution logged** and compared against the development
      distribution. If real capture quality falls outside the severity range the
      model trained on, the uncertainty estimates are extrapolating.
- [ ] **Deferred-case outcomes fed back.** Without knowing what the human decided,
      `human_rescue_rate` cannot be measured in production and the deferral
      threshold cannot be re-tuned.
- [ ] **A defined trigger for taking the model offline**, agreed in advance and
      owned by someone other than the person operating it.

## D. Governance and data

- [ ] Regulatory pathway identified for the jurisdiction of use. Nothing in this
      repository has regulatory clearance anywhere.
- [ ] Local ethics/IRB approval for prospective use, and a patient-consent
      position for images captured on personal phones.
- [ ] Image handling defined: a chest X-ray photograph on a clinician's phone is
      identifiable health data. Storage, transmission, retention, and deletion
      need answers before the first capture, not after.
- [ ] Dataset licences honoured for anything published — in particular the NIAID
      TB Portals DUA, which the aggregated Kaggle mirror does not visibly carry
      forward. See `LICENSE` and `docs/phase1_framing.md` section 4.
- [ ] **Failure modes disclosed to users in the interface**, including that the
      model was developed on synthetic degradation and validated on two held-out
      clinics of 138 and 662 images.

## E. Evidence that does not exist yet

Listed separately because these are not checkboxes a careful engineer can tick —
they require studies nobody has run:

- Prospective evaluation at the deployment site.
- Agreement between model uncertainty and radiologist-flagged ambiguity
  (`docs/LIMITATIONS.md` §4).
- Evidence that the deferral workflow improves patient outcomes rather than
  merely improving retained-set accuracy. A model can post excellent selective
  accuracy while making a clinic slower and no more accurate overall.
