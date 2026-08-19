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
      `ambiguous`, and `physics/triage.py` decides between them from the measured
      capture channel rather than from a confidence score — a high density floor
      caused by localized glare is operator-fixable, an adequate floor with a
      still-uncertain model is not.
- [ ] **The retake instruction says what to change.** "The image is bad" wastes
      the retake. `physics/triage.py` emits the specific fix — which way the glare
      is, whether the blur is shake or defocus, whether the exposure is crushed —
      from `glare.hotspot` and `psf.PSFEstimate.anisotropy`.

## B2. Two cheap additions to the lightbox

Both aim to turn an assumption in `physics/` into a measurement, and both cost almost
nothing. Both have now been measured rather than assumed
(`scripts/measure_fiducial_value.py`, 216 photographs): the ruler delivers, the wedge
does not yet, and neither changes a certificate verdict — at this severity range the
floor is dominated by the veil, so improving γ or the scale moves the inputs without
moving the decision.

- [ ] **Tape a small step wedge beside the film.** A few pence of exposed,
      processed film with two or three known optical densities, or about £20 for a
      calibrated 21-step wedge. With only the two anchors the film itself provides,
      the tone curve's exponent γ has to be taken from an sRGB prior and its
      uncertainty propagated into every absolute density. A third distinct density
      breaks that degeneracy and `physics/tone.fit_tone` fits γ properly.

      **Measured, and the measurement does not support the claim yet.**
      `scripts/measure_fiducial_value.py` runs the same photograph through the
      inversion with and without each aid, over 216 photographs. The fitted γ came
      out *worse* than the sRGB prior on average (0.65 vs 0.40) and its error bar
      covered the truth less often; it wins only where the ISP's real exponent is
      far above 2.2. Full numbers and the two candidate causes — the ISP's
      unmodelled contrast S-curve, and the veil where the wedge sits, outside the
      field, where the veil surface is extrapolated — are in
      [`PHYSICS.md`](PHYSICS.md) § "What the two cheap fiducials buy".

      So: still worth the pennies, but **do not quote the fitted γ or its error
      bar** until a wedge inside the collimated field has been tried. Use it with
      `physics.invert(..., aids=("wedge",))`; the strip is found by looking for the
      one monotone staircase outside the field.
- [ ] **Put a ruler, or any object of known length, in the frame.** `px_per_mm` is
      otherwise inferred from the detected collimation field against a standard
      cassette diagonal, at roughly ±20%. That error propagates into every
      finding's spatial frequency and so into the density floor. A ruler settles
      it — measured, 7.1% scale error down to 1.3%, at every true γ and every
      collimation tightness tested, which makes this the better of the two asks
      by a wide margin. `physics.invert(..., aids=True)` reads the tick *pitch* — not the
      ruler's length, which is easy to crop and hard to photograph squarely — and
      a train whose intervals are irregular is rejected rather than believed,
      because a scale error is silent and changes the floor with nothing anywhere
      saying so. You can still pass a value you measured yourself to
      `physics.invert(px_per_mm=...)`.

- [ ] **If either aid is in frame, expect the frame to be wider — and say so.**
      A photograph framed to include the lightbox beside the sheet has *bare
      lightbox* in it, which transmits everything against the film margin's 63%.
      `fiducials.detect` takes the base-fog anchor from everything outside the
      collimated field, so that anchor — the one the whole density scale hangs
      from — is biased bright by however much lightbox got in. `aids.margin_band`
      restricts it to the ring of film just outside the border, and `invert`
      applies that whenever `aids` is passed at all, including `aids=()`.
- [ ] **Train operators to photograph the *whole sheet*.** Including the pale
      unexposed margin, the L/R lead marker and the dark direct-exposure rim. Those
      three regions are the entire basis of the physics track, and a tight crop to
      the lung fields destroys all of them — after which the certificate can only
      abstain. `scripts/audit_fiducials.py` measures how often this is being done.
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
