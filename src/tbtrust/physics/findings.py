"""The signal side of the bound: what density contrast a TB finding actually presents.

The resolution floor in `floor.py` says what the photograph *can* carry. That is
only half a certificate. The other half is what the finding *is*: a lesion has a
characteristic optical-density contrast against the lung field around it and a
characteristic size, and the certificate is the comparison of the two.

Read this section before quoting a number
-----------------------------------------
`delta_d` is no longer a free-floating placeholder. As of 2026-08-16 every
contrast in the table is **computed** from a one-line radiographic-physics model
whose every input is a published constant, and each entry carries
`source="DERIVED-SLAB-v1"`. What that buys and what it does not:

    ΔD  =  (γ · C_s / ln 10) · (μ/ρ) · Δρ · t                        (the model)

* `γ` -- film average gradient, the slope of the H&D curve, so ΔD = γ·Δlog₁₀E.
* `C_s` -- scatter contrast-degradation factor, C_s = 1/(1 + SPR).
* `(μ/ρ)` -- mass attenuation coefficient at the beam's effective energy.
* `Δρ` -- the density step the finding makes against the lung it displaces.
* `t` -- the finding's thickness along the beam, taken as `size_mm`.

This is Beer-Lambert through a homogeneous slab, composed with the film's
characteristic curve. It is exactly the chain `density.py`'s docstring already
sets out, evaluated with numbers instead of left symbolic. The constants and
their provenance are in the `# derivation constants` block below; the sources are
NIST XCOM (mass attenuation coefficients), ICRU Report 44 (tissue densities), a
textbook value for film gradient, and a measured chest scatter-to-primary ratio.

What the derivation produces, against the placeholders it replaced::

    finding            derived ΔD    sigma    was (NOMINAL)
    miliary_nodule          0.018    0.008    0.030
    small_nodule            0.053    0.030    0.060
    infiltrate              0.066    0.048    0.090
    cavity_wall             0.040    0.023    0.220   <- largest move
    consolidation           0.398    0.203    0.350
    pleural_effusion        0.246    0.138    0.400
    fibrotic_band           0.044    0.025    0.080

Six of seven come out *lower* than the placeholders, so the certificate is
harsher, not more permissive, under the derivation. `cavity_wall` moves by 5.5x:
the placeholder had it as the second-highest contrast in the table on the reasoning
that a cavity is a high-contrast finding, but 3.4mm of wall is very little path,
and path length is what the exponent actually sees. Anything downstream that was
tuned against the old numbers -- the resolution sweep in `docs/PHYSICS.md` §3 in
particular -- needs re-running before it is quoted.

**What this is not.** It is a transcription-and-calculation, not a phantom
measurement, and the model is deliberately crude: it assumes a monoenergetic
effective beam, a homogeneous lesion, no beam hardening, and that everything
outside the lesion is common-mode and cancels in the difference. Two of the five
inputs (`γ`, `C_s`) are equipment-dependent and are carried as ranges, not point
values. So `delta_d` is now *traceable* rather than *invented* -- an auditable
number with an error bar -- and `delta_d_sigma` is the propagated uncertainty of
the model inputs together with the spread of presentations in `size_sigma_mm`.
A CDRAD-style phantom exposure on the clinic's own film/processor combination,
read with a densitometer, would still beat this and remains the right thing to do
before publishing an absolute verdict; `load_findings` takes that table with no
code change. See "How much to trust the absolute verdicts" at the end.

**`size_mm` / `size_sigma_mm` are cited from real literature for `miliary_nodule`,
`small_nodule`, and `cavity_wall`** -- see each entry's `note` for the specific
paper and caveats (in particular, `cavity_wall`'s 3.4mm uses the *thinnest*-wall
statistic from a CT study, not plain-film-measured). `infiltrate`, `consolidation`,
`pleural_effusion`, and
`fibrotic_band` remain unsourced placeholders: no citable single characteristic
diameter was found for these more diffuse, anatomically-variable findings.

Any publication using this code must still replace the remaining unsourced
`size_mm` fields, by either:

* transcribing size ranges from a radiographic reference or a contrast-detail
  study, and recording the citation in `source`; or
* measuring them directly -- expose a contrast-detail phantom (a CDRAD-style plate
  or, more cheaply, a step wedge with drilled cavities) on the same film/processor
  combination the clinic uses, and read the densities off with a densitometer.
  That same exposure replaces the derived `delta_d` with a measured one, which is
  the better instrument even where the derivation is defensible.

`load_findings` reads a YAML or CSV table so the swap needs no code change at all.

How much to trust the absolute verdicts
---------------------------------------
The certificate's *relative* statements -- this photo carries less density
resolution than that one, this glare hotspot costs you a factor of three, the
floor rose above the contrast when the veil passed 15% -- are sound, because they
depend only on the floor. They were sound under the old placeholders and are
unchanged by this derivation.

The *absolute* verdicts (`DETECTABLE` vs `INSUFFICIENT` for a named finding) now
rest on a traceable number rather than an invented one, but they inherit its
error bar, which is not small: the three shared constants alone put ≈24%
relative on every contrast, before either the per-finding density step or the
spread of presentations is added, and the whole table currently lands between 47%
and 73% relative. Worse, the two equipment constants (`γ`, `C_s`) shift *all*
findings together, so a systematic error there moves every verdict in the same
direction rather than averaging out across the rows of one certificate.
`FindingSpec.delta_d_sigma` is what carries this into the reported margin. Quote
a verdict with its margin and sigma, never bare.

Sign convention: `delta_d` is a magnitude. A consolidation is more attenuating
than the lung it replaces, so it *lowers* density; a cavity's air-filled lumen
raises it. Detection depends on the magnitude of the step, so the table stores
magnitudes and `film.insert_lesion` owns the sign.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# derivation constants -- every one of these is a published number, with the
# source next to it. Ranges are (low, high) and are propagated as uniform
# distributions (sigma = range / sqrt(12)) into `delta_d_sigma`.
# --------------------------------------------------------------------------- #

# Mass attenuation coefficient of soft tissue at the effective energy of a chest
# beam. NIST X-Ray Mass Attenuation Coefficients, Hubbell J.H. and Seltzer S.M.,
# NIST Standard Reference Database 126, table for "Tissue, Soft (ICRU-44)":
# 0.2264 cm^2/g at 50 keV, 0.2048 at 60 keV, 0.1823 at 80 keV, 0.1693 at 100 keV
# (https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/tissue.html).
# A 100-125 kVp chest technique has an effective energy in the 50-100 keV band;
# rather than commit to one effective energy we take the mid-band value and carry
# the whole band as the uncertainty. Lung tissue's coefficients are the same to
# within 0.3% over this band (.../ComTab/lung.html) -- the contrast comes from
# the *density* step, not from a composition difference, which is why the model
# below factors as (mu/rho) * Delta-rho.
MU_OVER_RHO_CM2_G = 0.19
MU_OVER_RHO_RANGE = (0.169, 0.227)

# Densities, g/cm^3. ICRU Report 44, "Tissue Substitutes in Radiation Dosimetry
# and Measurement" (ICRU, 1989): inflated lung 0.26, soft tissue 1.06. Water and
# air from NIST table 2 (1.000 and 1.205e-3). NOTE: NIST's "Lung Tissue (ICRU-44)"
# row lists 1.05 g/cm^3 -- that is the *composition* at tissue density, not an
# inflated lung. Using 1.05 here would erase the contrast entirely; 0.26 is the
# inflated-lung figure and the one this model needs.
RHO_LUNG_INFLATED = 0.26
RHO_LUNG_RANGE = (0.20, 0.40)      # inflation state varies within and across patients
RHO_SOFT_TISSUE = 1.06
RHO_PLEURAL_FLUID = 1.00           # transudate/exudate, taken as water
RHO_AIR = 0.001205

# Film average gradient (slope of the H&D curve between D = 0.25 and D = 2.0 above
# base+fog), dimensionless. "Radiographic films have gradients of ~2", with
# wide-latitude films -- the ones chosen for chest work precisely because the
# lung-to-mediastinum air-kerma range is so wide -- at the low end of the spread
# and high-contrast mammographic films above 3 (Radiology Key, "Image Quality",
# https://radiologykey.com/image-quality-2/). This is the least well-pinned
# constant in the chain and it scales every contrast in the table linearly.
FILM_AVG_GRADIENT = 2.0
FILM_AVG_GRADIENT_RANGE = (1.6, 2.8)

# Scatter contrast-degradation factor, C_s = 1/(1 + SPR). Scattered photons carry
# no lesion signal but do add to the exposure, so subject contrast reaches the
# film divided by (1 + SPR). Ungridded thoracic imaging runs at SPR 2-4 (Alahmari
# et al., "Efficacy of the scatter correction algorithm in portable chest
# radiography", PMC9130995), which an anti-scatter grid cuts by roughly an order
# of magnitude -- grid scatter transmission is typically 0.05-0.20 (ibid.); see
# also Barnes G.T., "Contrast and scatter in x-ray imaging", RadioGraphics
# 1991;11(2):307-323, DOI 10.1148/radiographics.11.2.2028065, for the general
# treatment. That leaves a gridded chest exposure at a residual SPR of roughly
# 0.2-1.0, i.e. C_s in 0.50-0.83. CAVEAT: this residual range is our arithmetic
# on the two cited numbers, not a directly measured gridded-chest SPR, and it is
# the weakest link in the derivation after gamma.
SCATTER_CDF = 0.67
SCATTER_CDF_RANGE = (0.50, 0.83)

DERIVATION_SOURCE = "DERIVED-SLAB-v1"


def _uniform_sigma(rng: tuple[float, float]) -> float:
    """Standard deviation of a uniform distribution over `rng`."""
    lo, hi = rng
    return float((hi - lo) / np.sqrt(12.0))


# Relative uncertainty contributed by the three constants shared by every finding.
# These are *correlated across findings* -- gamma is a property of the film, not
# of the lesion -- so they inflate each entry's sigma but do not average out when
# several findings are read off one certificate.
_REL_SIGMA_SHARED = float(np.sqrt(
    (_uniform_sigma(FILM_AVG_GRADIENT_RANGE) / FILM_AVG_GRADIENT) ** 2
    + (_uniform_sigma(SCATTER_CDF_RANGE) / SCATTER_CDF) ** 2
    + (_uniform_sigma(MU_OVER_RHO_RANGE) / MU_OVER_RHO_CM2_G) ** 2
))


def slab_delta_d(delta_rho: float, size_mm: float) -> float:
    """Film density contrast of a slab of density step `delta_rho`, `size_mm` thick.

    ΔD = (gamma * C_s / ln 10) * (mu/rho) * Delta-rho * t, with t in cm.
    """
    t_cm = float(size_mm) / 10.0
    return float(FILM_AVG_GRADIENT * SCATTER_CDF / np.log(10.0)
                 * MU_OVER_RHO_CM2_G * float(delta_rho) * t_cm)


def slab_delta_d_sigma(delta_rho: float, delta_rho_sigma: float,
                       size_mm: float, size_sigma_mm: float) -> float:
    """Propagated 1-sigma on `slab_delta_d`.

    The model is a product, so relative uncertainties add in quadrature: the three
    shared constants, the density step, and the spread of presentations already
    recorded in `size_sigma_mm`. The last term is why a finding with a wide size
    distribution reports a wide contrast even where the physics is well pinned.
    """
    rel = np.sqrt(_REL_SIGMA_SHARED ** 2
                  + (float(delta_rho_sigma) / float(delta_rho)) ** 2
                  + (float(size_sigma_mm) / float(size_mm)) ** 2)
    return float(rel * slab_delta_d(delta_rho, size_mm))


@dataclass(frozen=True)
class FindingSpec:
    """One radiographic finding, as a detectability target."""

    key: str
    name: str
    delta_d: float              # |density contrast| against adjacent lung, magnitude
    delta_d_sigma: float        # spread across presentations; enters the margin
    size_mm: float              # characteristic diameter or wall thickness
    size_sigma_mm: float
    note: str = ""
    source: str = "NOMINAL"

    def size_px(self, px_per_mm: float) -> float:
        return float(self.size_mm * px_per_mm)

    def char_frequency(self, px_per_mm: float) -> float:
        """Characteristic spatial frequency in cycles/pixel.

        A feature of width d contributes most of its energy near f = 1/(2d) --
        the fundamental of a bar pair at that width. Used only for the
        interpretable single-number MTF diagnostic; the floor itself integrates
        the whole template spectrum in `floor.template_energy`, which needs no
        such choice.
        """
        d = max(self.size_px(px_per_mm), 1e-6)
        return float(1.0 / (2.0 * d))

    def rescaled(self, delta_d: float | None = None, size_mm: float | None = None) -> FindingSpec:
        return replace(self,
                       delta_d=self.delta_d if delta_d is None else float(delta_d),
                       size_mm=self.size_mm if size_mm is None else float(size_mm))


def _derived(key: str, name: str, delta_rho: float, delta_rho_sigma: float,
             size_mm: float, size_sigma_mm: float, note: str = "") -> FindingSpec:
    """Build a spec whose contrast comes from the slab model, not from a guess.

    `delta_rho` is the density step the finding makes against the lung it
    displaces, in g/cm^3, and is the one modelling choice left per finding: which
    two materials the lesion puts on either side of the step. Each call documents
    that choice in its note.
    """
    return FindingSpec(
        key, name,
        delta_d=slab_delta_d(delta_rho, size_mm),
        delta_d_sigma=slab_delta_d_sigma(delta_rho, delta_rho_sigma, size_mm, size_sigma_mm),
        size_mm=size_mm, size_sigma_mm=size_sigma_mm,
        note=note, source=DERIVATION_SOURCE,
    )


# --------------------------------------------------------------------------- #
# the table -- READ THE MODULE DOCSTRING BEFORE PUBLISHING THESE
#
# delta_d is computed, not typed in: change a constant above and every row moves
# together, which is the point. size_mm is still per-finding evidence and several
# entries are still unsourced there -- each note says which.
# --------------------------------------------------------------------------- #
TB_FINDINGS: dict[str, FindingSpec] = {
    f.key: f
    for f in (
        _derived(
            "miliary_nodule", "Miliary nodule",
            # granulomatous nodule (soft tissue) displacing inflated lung
            RHO_SOFT_TISSUE - RHO_LUNG_INFLATED, _uniform_sigma(RHO_LUNG_RANGE),
            2.0, 0.8,
            note="The hardest target by a wide margin: small and low contrast at once. "
                 "If any finding is going to fall below the floor of a phone photo, it is this one. "
                 "size_mm is cited, delta_d is not: nodules measure <3mm in >90% of correctly "
                 "identified miliary TB cases, typically 1-2mm (Kwong et al., 'Miliary tuberculosis. "
                 "Diagnostic accuracy of chest radiography', PubMed 8697830); STATPEARLS NBK562300 "
                 "gives the same 1-2mm range. delta_d is derived, not measured: a solid granuloma "
                 "replacing inflated lung is a soft-tissue-vs-lung density step (1.06 - 0.26 g/cm^3) "
                 "over 2mm of path. It stays the hardest target under the derivation -- the derived "
                 "contrast is *below* the old placeholder, so if anything the miliary verdict is now "
                 "harsher. No contrast-detail phantom study reporting film optical density by TB "
                 "finding type was found (searches 2026-08-15 and 2026-08-16), so this is a "
                 "calculation from published constants, not a transcription of a measured value.",
        ),
        _derived(
            "small_nodule", "Small nodule",
            # same substitution as the miliary nodule, larger
            RHO_SOFT_TISSUE - RHO_LUNG_INFLATED, _uniform_sigma(RHO_LUNG_RANGE),
            6.0, 3.0,
            note="Individual granuloma or a small focus of disease. delta_d is derived from the same "
                 "soft-tissue-vs-inflated-lung step as miliary_nodule, over 6mm instead of 2mm. "
                 "size_mm sits in the standard "
                 "'subcentimeter nodule' bracket (<10mm) and specifically the <6mm bracket radiology "
                 "literature associates with the lowest malignancy probability (~1%, rising to "
                 "~10-20% around 8mm), distinguishing it from the <3mm 'micronodule'/miliary bracket, "
                 "per MacMahon H et al., 'Guidelines for Management of Incidental Pulmonary Nodules "
                 "Detected on CT Images: From the Fleischner Society 2017,' Radiology 2017; "
                 "284(1):228-243, DOI 10.1148/radiol.2017161659. This is a size-category citation from "
                 "CT-nodule-management literature, not a TB-granuloma-specific or plain-film-specific "
                 "measurement. (An earlier version of this "
                 "note cited an AJR paper for this claim without confirming it was the actual source -- "
                 "corrected 2026-08-15 after the Fleischner 2017 guideline was verified directly.)",
        ),
        _derived(
            "infiltrate", "Patchy infiltrate",
            # PARTIAL airspace filling: half the alveolar air replaced by exudate,
            # so half the full consolidation step. The 0.5 fill fraction is an
            # assumption, not a citation; its sigma is set wide to say so.
            0.5 * (RHO_SOFT_TISSUE - RHO_LUNG_INFLATED), 0.20,
            15.0, 7.0,
            note="Ill-defined airspace opacity; the low-contrast edge is what makes it hard, "
                 "not the size. delta_d is derived, but with the weakest input in the table: a patchy "
                 "infiltrate only partially fills the airspace, and the fill fraction used here (0.5, "
                 "i.e. half the soft-tissue-vs-lung step) is an assumption with no citation behind it. "
                 "Its uncertainty is set to +/-0.20 g/cm^3 -- half the step itself -- so the derived "
                 "contrast for this row carries roughly 50% relative error before size spread. No "
                 "specific citable linear-mm size was found for this finding either -- "
                 "'patchy infiltrate' extent varies too widely across presentations for a single "
                 "characteristic diameter to be meaningful, and no source measuring one was located "
                 "in this session, so size_mm remains an unsourced placeholder.",
        ),
        _derived(
            "cavity_wall", "Cavity wall",
            # the wall is soft tissue and the lumen it bounds is air, so this is the
            # largest density step of any finding here -- but over the thinnest path
            RHO_SOFT_TISSUE - RHO_AIR, _uniform_sigma((0.9, 1.15)),
            3.4, 1.8,
            note="High contrast but thin. Blur-limited rather than contrast-limited, which is "
                 "why it behaves quite differently from an infiltrate under motion. size_mm/size_sigma_mm "
                 "updated from an earlier placeholder (4.0/2.0) to a cited value, chosen carefully: "
                 "Kim et al., 'Comparison of chest CT findings in nontuberculous mycobacterial diseases "
                 "vs. Mycobacterium tuberculosis lung disease' (PMC5367717) reports TWO wall-thickness "
                 "statistics per cavity -- thickest (10.9+/-5.6mm TB, 6.9+/-3.7mm NTM) and thinnest "
                 "(3.4+/-1.8mm TB, 2.8+/-1.2mm NTM), Table 2. This entry uses the THINNEST-wall figure, "
                 "because a cavity's thinnest segment is the blur-limited dimension this note already "
                 "describes -- the thickest segment is not what a resolution floor needs to clear. "
                 "Caveat: CT-measured, not plain-film-measured -- the wall thickness actually resolvable "
                 "on a projection radiograph/phone photo may differ from the CT cross-sectional value. "
                 "delta_d is derived from the wall-vs-lumen step (soft tissue against air, the biggest "
                 "density step in the table) over that same 3.4mm, which is the modelling choice worth "
                 "arguing with: it is the step across the *inner* boundary of the wall. Against the "
                 "surrounding lung rather than the lumen the step would be ~25% smaller. Note this "
                 "row's contrast falls well below the old 0.220 placeholder, because the placeholder "
                 "was not accounting for how little path 3.4mm of wall actually is.",
        ),
        _derived(
            "consolidation", "Lobar consolidation",
            # complete airspace filling: alveolar air fully replaced by exudate
            RHO_SOFT_TISSUE - RHO_LUNG_INFLATED, _uniform_sigma(RHO_LUNG_RANGE),
            45.0, 20.0,
            note="Large and high contrast. Survives almost any capture; useful as the control "
                 "that shows the certificate is not simply failing everything. delta_d is derived "
                 "assuming complete airspace filling -- the full soft-tissue-vs-inflated-lung step -- "
                 "over the whole 45mm depth, which is what makes it an order of magnitude above the "
                 "nodules. No specific citable "
                 "linear-mm size was found -- lobar consolidation extent is defined by anatomy (a "
                 "lobe), not a characteristic lesion diameter, and no source giving one was located "
                 "in this session, so size_mm remains an unsourced placeholder and it enters delta_d "
                 "directly: this row's contrast is only as good as that 45mm.",
        ),
        _derived(
            "pleural_effusion", "Pleural effusion",
            # fluid (taken as water) displacing inflated lung
            RHO_PLEURAL_FLUID - RHO_LUNG_INFLATED, _uniform_sigma(RHO_LUNG_RANGE),
            30.0, 15.0,
            note="Blunted costophrenic angle. Peripheral, so it is the finding most often lost "
                 "to a crop or to vignetting rather than to noise. Related citable data exists but "
                 "was not converted into size_mm: erect-PA blunting is detectable from ~200mL, per "
                 "Moskowitz H, Platt RT, Schachar R, et al., 'Roentgen visualization of minute pleural "
                 "effusion,' Radiology 1973;109(1):33-35 (lateral-view blunting is more sensitive, "
                 "~75mL, per secondary sources citing the same study -- the original was not directly "
                 "confirmed for that specific number). This is a volume, not a linear dimension, and "
                 "converting it to a 'characteristic diameter' would need a geometric assumption not "
                 "attempted here, so size_mm remains an unsourced placeholder. (An "
                 "earlier version of this note miscited an AJR paper about *supine* radiographs for "
                 "this *erect*-view number -- corrected 2026-08-15.) delta_d is derived from fluid "
                 "(taken as water, 1.00 g/cm^3) displacing inflated lung over that 30mm depth.",
        ),
        _derived(
            "fibrotic_band", "Fibrotic band / scarring",
            # fibrous tissue in place of inflated lung; taken as soft tissue, with a
            # wider density sigma because mature scar is denser and may calcify
            RHO_SOFT_TISSUE - RHO_LUNG_INFLATED, 0.12,
            5.0, 2.5,
            note="Post-primary sequela; thin and moderate contrast. delta_d is derived from a "
                 "fibrous-tissue-vs-lung step, taken as soft tissue with a widened density sigma "
                 "(0.12 rather than the 0.058 the lung-inflation range alone gives) because mature "
                 "scar is denser than generic soft tissue and may calcify, which would push the step "
                 "up. No specific citable linear-mm size was found in this session, so size_mm "
                 "remains an unsourced placeholder.",
        ),
    )
}

# The subset worth reporting when you only have room for a few rows: the hardest
# target, a mid target, and a control that should always pass.
CORE_FINDINGS = ("miliary_nodule", "infiltrate", "cavity_wall", "consolidation")


def get(key: str) -> FindingSpec:
    if key not in TB_FINDINGS:
        raise KeyError(f"unknown finding {key!r}; known: {sorted(TB_FINDINGS)}")
    return TB_FINDINGS[key]


def core() -> list[FindingSpec]:
    return [TB_FINDINGS[k] for k in CORE_FINDINGS]


def all_findings() -> list[FindingSpec]:
    return list(TB_FINDINGS.values())


def load_findings(path: str | Path) -> dict[str, FindingSpec]:
    """Replace the nominal table from a YAML or CSV file.

    Expected columns / keys: key, name, delta_d, delta_d_sigma, size_mm,
    size_sigma_mm, and optionally note and source. `source` should say where the
    number came from -- it is printed in the certificate provenance block, so a
    reader can see at a glance whether the verdict rests on a measurement or on
    the placeholder.
    """
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        rows = yaml.safe_load(p.read_text())
        rows = rows["findings"] if isinstance(rows, dict) and "findings" in rows else rows
    else:
        import csv

        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))

    out: dict[str, FindingSpec] = {}
    for r in rows:
        out[str(r["key"])] = FindingSpec(
            key=str(r["key"]),
            name=str(r.get("name", r["key"])),
            delta_d=float(r["delta_d"]),
            delta_d_sigma=float(r.get("delta_d_sigma", 0.0)),
            size_mm=float(r["size_mm"]),
            size_sigma_mm=float(r.get("size_sigma_mm", 0.0)),
            note=str(r.get("note", "")),
            source=str(r.get("source", "user-supplied")),
        )
    if not out:
        raise ValueError(f"no findings parsed from {path}")
    return out


def install(findings: dict[str, FindingSpec]) -> None:
    """Swap the module-level table in place, so every consumer picks it up."""
    TB_FINDINGS.clear()
    TB_FINDINGS.update(findings)


def contrast_detail_curve(findings: list[FindingSpec] | None = None) -> np.ndarray:
    """The (size_mm, delta_d) points as a contrast-detail plot, for figures.

    Overlaying the measured floor on this is the single clearest picture the
    physics track produces: findings above the floor curve are carried by the
    photograph, findings below it are not, and the gap between a clean capture's
    floor and a glared one's is visible at a glance.
    """
    fs = findings or all_findings()
    return np.array([[f.size_mm, f.delta_d] for f in fs], dtype=np.float64)
