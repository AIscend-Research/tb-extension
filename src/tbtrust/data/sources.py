"""Candidate CXR sources, and what each would actually buy this rotation.

The rotation reports two honest holdout folds. `docs/LIMITATIONS.md` §3 is blunt
about why: the manifest draws on four sources but only Montgomery and Shenzhen
contain both classes, and a single-class holdout has an undefined sensitivity or
specificity and a clinic label that is a near-perfect proxy for the diagnosis.
Two folds is a thin basis for a cross-site claim, and no modelling change fixes
it -- a third and fourth two-class site would strengthen the claim more than any
loss function could.

This module is the survey, in a form code can read: what exists, what it costs
to get, and -- the part that decides everything -- whether it is genuinely an
*independent site* rather than a remix of the sites already in the manifest.

**The trap this registry exists to document.** The most convenient TB dataset on
Kaggle, the Qatar/Dhaka "Tuberculosis (TB) Chest X-ray Database", is the one
`scripts/download_data.py --kaggle-aggregated` fetches. It is not a new clinic.
It is a *re-bundling* of NLM, Belarus, NIAID and RSNA, so adding it as a fifth
clinic would put Shenzhen images on both sides of a leave-one-clinic-out split
while the split code, which keys on the clinic label, reported a clean fold.
That is worse than having two folds: it is having four that quietly lie.
`scripts/audit_source_overlap.py` is the check that catches it on the pixels
rather than on trust.

Every field below is marked with how it was established. `verified` means it was
read off a primary source (the dataset's own page, its paper, or its licence
text) while compiling this; `reported` means it comes from secondary literature
and should be re-checked against the download; `unknown` means nobody has
checked. Counts in particular have a way of drifting between a paper's table and
the tarball that actually arrives, so `scripts/build_manifest.py` re-derives the
class balance from the files rather than trusting anything here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Sources already in the manifest, for reference in the tables below.
IN_MANIFEST = ("montgomery", "shenzhen", "niaid", "rsna", "belarus")


@dataclass(frozen=True)
class Source:
    """One candidate source and the facts that decide whether it can be a fold."""

    key: str
    name: str
    country: str
    institution: str
    n_images: int | None
    both_classes: bool | None
    class_note: str
    native_resolution: str
    fmt: str
    licence: str
    access: str                       # what a human must do to get it
    url: str
    independent_of: tuple[str, ...]   # sites it is known NOT to overlap
    overlap_risk: str
    verdict: str                      # usable | usable_with_caveats | not_a_fold | blocked
    why: str
    evidence: dict = field(default_factory=dict)   # field -> verified|reported|unknown

    @property
    def is_candidate_fold(self) -> bool:
        return self.verdict in {"usable", "usable_with_caveats"}


#: Ranked by what they buy this project, not by size. The ranking is the point:
#: the biggest dataset here is not the most useful one, because a source
#: pre-downscaled to 512 px cannot carry the physics track's fiducials.
CANDIDATES: tuple[Source, ...] = (
    Source(
        key="nitrd_da",
        name="NITRD set DA",
        country="India",
        institution="National Institute of TB and Respiratory Diseases, New Delhi",
        n_images=156,
        both_classes=True,
        class_note="~78 abnormal / ~75 normal; DA and DB were shot on different "
                   "X-ray machines at the same institute",
        native_resolution="varies; full-resolution radiographs",
        fmt="png/jpg",
        licence="research use; redistributed on Kaggle",
        access="Kaggle download, no gate",
        url="https://www.kaggle.com/datasets/vbookshelf/da-and-db-tb-chest-x-ray-datasets",
        independent_of=("montgomery", "shenzhen", "rsna"),
        overlap_risk="low; predates and is disjoint from the NLM sets, but it is "
                     "bundled into TBX11K's comparison tables, so check before "
                     "using both",
        verdict="usable",
        why="The best value here by a distance. Roughly Montgomery-sized, "
            "genuinely two-class, a third country and population, no access gate, "
            "and -- unusually -- DA and DB are two different machines at one "
            "institute, which is the within-site/between-machine shift this "
            "project keeps assuming and has never been able to measure.",
        evidence={"n_images": "reported", "both_classes": "reported",
                  "institution": "reported", "licence": "unknown"},
    ),
    Source(
        key="nitrd_db",
        name="NITRD set DB",
        country="India",
        institution="National Institute of TB and Respiratory Diseases, New Delhi",
        n_images=150,
        both_classes=True,
        class_note="~78 abnormal / ~75 normal; sibling of DA on a different machine",
        native_resolution="varies; full-resolution radiographs",
        fmt="png/jpg",
        licence="research use; redistributed on Kaggle",
        access="Kaggle download, no gate",
        url="https://www.kaggle.com/datasets/vbookshelf/da-and-db-tb-chest-x-ray-datasets",
        independent_of=("montgomery", "shenzhen", "rsna"),
        overlap_risk="low, but DA and DB share an institute -- treat them as two "
                     "machines at one site, not two independent sites, unless the "
                     "clinic-stats table says otherwise",
        verdict="usable",
        why="See DA. Taken together they take the rotation from two folds to four, "
            "which is the single largest available improvement to the cross-site "
            "claim.",
        evidence={"n_images": "reported", "both_classes": "reported",
                  "institution": "reported", "licence": "unknown"},
    ),
    Source(
        key="tbx11k",
        name="TBX11K",
        country="China",
        institution="undisclosed hospitals (Liu et al., CVPR 2020 / TPAMI)",
        n_images=11200,
        both_classes=True,
        class_note="5000 healthy, 5000 sick-but-non-TB, 924 active TB, 212 latent, "
                   "54 both, 10 uncertain; only train+val ground truth is public "
                   "(~8976 images), the test set is held for a challenge",
        native_resolution="distributed at 512x512, downscaled from ~3000x3000",
        fmt="png",
        licence="CC BY 4.0",
        access="Google Drive / Baidu link from the project page",
        url="https://github.com/yun-liu/Tuberculosis",
        independent_of=(),
        overlap_risk="stated to be a new collection rather than a compilation, but "
                     "it is Chinese and the Shenzhen set is Chinese; secondary "
                     "sources conflate the two often enough that the pixel-level "
                     "audit is mandatory before it goes in the manifest",
        verdict="usable_with_caveats",
        why="Far the largest, permissively licensed, and it has both classes. Two "
            "caveats decide how it is used. It ships at 512x512, downscaled from "
            "~3000x3000 -- so it can serve the classifier and the cross-site "
            "claim, but *not* the physics track, whose fiducial detection and PSF "
            "estimate need the resolution the downscale threw away. And the "
            "'sick-but-non-TB' class makes it the only source here that can "
            "separate 'abnormal' from 'TB', which is a different and harder "
            "question than the one this project currently asks.",
        evidence={"n_images": "verified", "class_note": "verified",
                  "licence": "verified", "native_resolution": "verified",
                  "institution": "unknown"},
    ),
    Source(
        key="vindr_cxr",
        name="VinDr-CXR",
        country="Vietnam",
        institution="Hospital 108 and Hanoi Medical University Hospital",
        n_images=18000,
        both_classes=True,
        class_note="Tuberculosis is one of six global labels (with lung tumour, "
                   "pneumonia, COPD, other diseases, no finding); the per-label "
                   "TB count is not stated on the landing page and must be read "
                   "off the annotation CSV",
        native_resolution="original DICOM, full resolution",
        fmt="DICOM",
        licence="PhysioNet Credentialed Health Data License 1.5.0",
        access="credentialed PhysioNet account + CITI training + signed DUA",
        url="https://physionet.org/content/vindr-cxr/1.0.0/",
        independent_of=("montgomery", "shenzhen", "niaid", "rsna", "belarus"),
        overlap_risk="none known; a separate country and hospital system",
        verdict="usable_with_caveats",
        why="Radiologist-labelled, full-resolution DICOM from two named hospitals "
            "in a fourth country -- on the merits the strongest addition, and the "
            "only candidate besides the NLM sets that keeps the physics track "
            "alive at native resolution. The caveat is lead time, not quality: "
            "CITI training plus a credentialed DUA is weeks, not an afternoon, so "
            "start it before it is needed rather than when.",
        evidence={"n_images": "verified", "class_note": "verified",
                  "licence": "verified", "institution": "verified",
                  "both_classes": "verified"},
    ),
    Source(
        key="padchest",
        name="PadChest",
        country="Spain",
        institution="Hospital Universitario de San Juan, Alicante",
        n_images=160000,
        both_classes=True,
        class_note="174 findings from reports; 'tuberculosis' and 'tuberculosis "
                   "sequelae' are among them, on the order of hundreds of studies "
                   "each -- the exact counts must be taken from the label CSV, and "
                   "only ~27% of reports were annotated by a physician, the rest "
                   "by a recurrent network over the report text",
        native_resolution="original, full resolution",
        fmt="png/DICOM",
        licence="research use, via BIMCV",
        access="BIMCV registration",
        url="https://bimcv.cipf.es/bimcv-projects/padchest/",
        independent_of=("montgomery", "shenzhen", "niaid", "rsna", "belarus"),
        overlap_risk="none known; European, a different population and disease "
                     "prevalence entirely",
        verdict="usable_with_caveats",
        why="A European site would test the cross-site claim harder than another "
            "Asian one, and the resolution suits the physics track. But the TB "
            "label is text-mined from reports for roughly three quarters of the "
            "set, which is a materially weaker label than the NLM sets' "
            "radiologist consensus -- and TB prevalence in Alicante is low enough "
            "that the positive class will be small and possibly dominated by "
            "sequelae rather than active disease. Usable as a fold, but the label "
            "provenance has to be stated next to the number.",
        evidence={"n_images": "verified", "institution": "verified",
                  "class_note": "reported"},
    ),
    Source(
        key="kaggle_tb_aggregate",
        name="Tuberculosis (TB) Chest X-ray Database (Qatar/Dhaka)",
        country="mixed",
        institution="re-bundled from NLM, Belarus, NIAID and RSNA",
        n_images=4200,
        both_classes=True,
        class_note="700 TB / 3500 normal -- but the TB comes mostly from NIAID and "
                   "Belarus and the normals mostly from RSNA",
        native_resolution="512x512, downscaled",
        fmt="png",
        licence="CC BY 4.0 on the bundle",
        access="Kaggle download, no gate",
        url="https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset",
        independent_of=(),
        overlap_risk="CERTAIN. It contains the NLM images already in this manifest.",
        verdict="not_a_fold",
        why="The most available and the most dangerous. It is a remix of sources "
            "already here, so adding it as a clinic would put the same images on "
            "both sides of a leave-one-clinic-out split while the split code "
            "reported a clean fold -- and its class balance is confounded with "
            "source, which is the exact failure docs/DATA.md warns about. Fine as "
            "a convenience mirror for the images it re-hosts. Never a fifth "
            "clinic.",
        evidence={"class_note": "reported", "overlap_risk": "verified"},
    ),
    Source(
        key="niaid_tbportals",
        name="NIAID TB Portals",
        country="multi (Belarus, Moldova, Georgia, ...)",
        institution="NIAID TB patient registry",
        n_images=3000,
        both_classes=False,
        class_note="a registry of TB patients: almost entirely positive",
        native_resolution="varies",
        fmt="varies",
        licence="DUA",
        access="click-through data-use agreement",
        url="https://data.tbportals.niaid.nih.gov/",
        independent_of=("montgomery", "shenzhen", "rsna"),
        overlap_risk="re-bundled inside the Kaggle aggregate above",
        verdict="not_a_fold",
        why="Already in the manifest and already excluded as a holdout for the "
            "right reason. Listed here so the registry is a complete account "
            "rather than only the good news.",
        evidence={"both_classes": "verified"},
    ),
)

BY_KEY = {s.key: s for s in CANDIDATES}


def candidate_folds() -> list[Source]:
    """Sources that could serve as an additional two-class holdout."""
    return [s for s in CANDIDATES if s.is_candidate_fold]


def summary_rows() -> list[dict]:
    """Flat rows for a table or a CSV."""
    return [{
        "key": s.key, "name": s.name, "country": s.country,
        "n_images": s.n_images, "both_classes": s.both_classes,
        "resolution": s.native_resolution, "licence": s.licence,
        "access": s.access, "verdict": s.verdict,
        "overlap_risk": s.overlap_risk, "url": s.url,
        "unverified_fields": sorted(k for k, v in s.evidence.items()
                                    if v != "verified"),
    } for s in CANDIDATES]
