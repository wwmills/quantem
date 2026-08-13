"""
``zbkit`` -- simulation side of the single-image drift-correction project.

Builds a zincblende <110> substrate with an epitaxial cubic film on top, images it
through an *uncorrected* STEM probe (so the dumbbells are only marginally resolved),
and applies a known scan drift in scan coordinates rather than by warping the finished
image.  The correction itself lives in ``quantem.imaging.drift_single_image``.
"""

from . import imaging, structures, style
from .imaging import (
    GaussianMixtureKernel,
    Probe,
    ScanDrift,
    add_poisson_noise,
    apply_mtf,
    ground_truth_positions,
    normalize,
    render,
    simulate,
)
from .structures import (
    Columns,
    cubic_film_100,
    epitaxial_stack,
    misfit_staircase,
    relaxing_dislocation_spacing,
    rocksalt_110,
    zincblende_110,
)

__all__ = [
    "imaging", "structures", "style",
    "Probe", "GaussianMixtureKernel", "ScanDrift", "render", "simulate",
    "ground_truth_positions", "apply_mtf", "add_poisson_noise", "normalize",
    "Columns", "zincblende_110", "rocksalt_110", "cubic_film_100",
    "epitaxial_stack", "misfit_staircase", "relaxing_dislocation_spacing",
]
