"""Optical density: the radiometrically meaningful quantity, and its conventions.

Why density and not pixel values
--------------------------------
Photographic optical density is defined as

    D = -log10(tau),      tau = transmitted / incident luminance

and for a screen-film radiograph D is, over the film's linear latitude,
proportional to the log of X-ray exposure, which by Beer-Lambert is linear in the
path-integrated attenuation coefficient of the patient:

    D  ~  gamma_film * log10(E)  ~  gamma_film * (-mu * t) / ln(10) + const

So a *difference* in density, `Delta D`, is proportional to a difference in
path-integrated attenuation. That is the quantity a radiologist is actually
reading, and it is the quantity whose smallest resolvable value we can bound.
Pixel values in a phone photo are that signal after a lightbox, a lens with
veiling glare, an unknown ISP tone curve and an 8-bit quantiser -- four
transformations, three of them unknown and one of them lossy. The whole point of
`physics/invert.py` is to undo them and get back to D.

Sign convention (this trips everyone up, including the project brief)
---------------------------------------------------------------------
There are two opposite conventions in play and they must not be mixed.

* **X-ray transmission.** Lead transmits nothing. In beam-stop terms the lead
  marker is "zero signal".
* **Developed film / display.** Film is a negative: *more* X-ray exposure means
  *more* silver means *higher* optical density means *darker* on a lightbox.
  Lead blocks the beam, so the film under a lead marker is barely exposed, so it
  is near base+fog density, so it is **transparent -- the brightest thing on the
  lightbox**. Conversely the direct-exposure region (beam straight to the film,
  no patient in the way) is at maximum density and is **the darkest thing**.

Everything downstream of the film -- which is everything in this package, since
we only ever see photographs of developed film -- uses the second convention:

    bright pixel  <->  low D   <->  low X-ray exposure  <->  bone, lead, collimated-out
    dark pixel    <->  high D  <->  high X-ray exposure <->  direct-exposure region, air

The project brief describes the lead marker as the black anchor and the
direct-exposure region as white. That is the transmission convention, and it is
inverted relative to what a camera pointed at a lightbox records. The idea is
unaffected -- there are still two anchors of known density and one of them is
optically black -- but the roles swap:

    optical beam stop / coronagraph  =  direct-exposure region  (D_MAX, near-opaque)
    bright densitometry anchor       =  lead marker             (D_MIN, near-clear)

`fiducials.py` finds both under those names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# film characteristic constants
# --------------------------------------------------------------------------- #
# Base-plus-fog density of processed medical film: the density of an unexposed,
# developed sheet. Nothing on the film is ever clearer than this, so it is the
# density under a lead marker and outside the collimation border.
D_MIN_DEFAULT = 0.20

# Maximum useful density of medical film. Direct-exposure regions saturate here.
# tau = 10^-3.2 ~ 6e-4: essentially opaque, which is what makes it usable as an
# optical beam stop.
D_MAX_DEFAULT = 3.20

# Density range a chest radiograph's *diagnostic* (lung field) region occupies.
# Used to sanity-check an inversion and to set the display mapping in film.py.
D_LUNG_RANGE = (0.35, 2.30)


@dataclass(frozen=True)
class FilmModel:
    """The two density anchors the fiducials pin, plus their tolerances.

    `d_min_sigma` / `d_max_sigma` are the honest admission that base+fog and
    D_max vary between film stocks, processors and how old the developer was.
    They propagate into the absolute density accuracy in `invert.py`. They do
    *not* enter the resolution floor in `floor.py`, because that bounds a
    *difference* of densities measured in one image, and a common offset or a
    common scale error cancels in a difference. Keeping those two error budgets
    separate is the difference between a defensible bound and a hand-wave.
    """

    d_min: float = D_MIN_DEFAULT
    d_max: float = D_MAX_DEFAULT
    d_min_sigma: float = 0.05
    d_max_sigma: float = 0.30

    def __post_init__(self):
        if not self.d_max > self.d_min:
            raise ValueError(f"d_max ({self.d_max}) must exceed d_min ({self.d_min})")

    @property
    def tau_min(self) -> float:
        """Transmittance of the *brightest* film region (the lead marker)."""
        return density_to_transmittance(self.d_min)

    @property
    def tau_max(self) -> float:
        """Transmittance of the *darkest* film region (direct exposure)."""
        return density_to_transmittance(self.d_max)

    @property
    def dynamic_range_db(self) -> float:
        """Optical dynamic range the film presents to the camera, in dB."""
        return float(20.0 * np.log10(self.tau_min / self.tau_max))


def density_to_transmittance(d):
    """tau = 10^-D."""
    return np.power(10.0, -np.asarray(d, dtype=np.float64))


def transmittance_to_density(tau, floor: float = 1e-8):
    """D = -log10(tau), with tau floored so a zeroed pixel gives a finite D."""
    t = np.maximum(np.asarray(tau, dtype=np.float64), floor)
    return -np.log10(t)


def display_to_density(x, film: FilmModel | None = None):
    """Map a normalised display image in [0, 1] (1 = white) onto optical density.

    This is the assumed transfer of the *digitised archive* -- the linear ramp
    from D_max at black to D_min at white that a scanner nominally applies. It
    is used by the forward model in `film.py` to turn a public dataset PNG back
    into a plausible density map so we have something physical to photograph.

    It is emphatically not used on the inverse path: `invert.py` never assumes
    the archive's transfer, it measures the *phone's* transfer from the fiducials.
    """
    film = film or FilmModel()
    v = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return film.d_max + (film.d_min - film.d_max) * v


def density_to_display(d, film: FilmModel | None = None):
    """Inverse of `display_to_density`, for showing a density map to a human."""
    film = film or FilmModel()
    dd = np.asarray(d, dtype=np.float64)
    return np.clip((film.d_max - dd) / (film.d_max - film.d_min), 0.0, 1.0)


def delta_density_from_contrast(background_d, contrast_ratio: float) -> np.ndarray:
    """Density step corresponding to a fractional transmittance change.

    A lesion that attenuates `contrast_ratio` more of the beam shifts density by
    log10(1 + contrast_ratio) independently of the background level -- density is
    a log quantity, which is exactly why lesion contrast is quotable as a single
    number per finding type in `findings.py` rather than as a function of where
    in the lung it sits.
    """
    _ = background_d
    return float(np.log10(1.0 + contrast_ratio))
