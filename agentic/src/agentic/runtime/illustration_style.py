from __future__ import annotations


PAPER_STORYBOOK_SUBJECT_DIRECTION = (
    "Default subject art direction: render the subject itself as a refined handcrafted illustration in the "
    "selected medium, with surface detail and mark quality suited to that medium, nuanced light and soft "
    "dimensional shadows, and a restrained yet clearly chromatic palette. Preserve the requested silhouette "
    "and any transparent, solid, or chroma-key "
    "background exactly; do not add a floor or environment when the asset requires isolation. Honor an "
    "explicitly requested different medium or palette."
)


def apply_isolated_subject_art_direction(prompt: str) -> str:
    """Add the cutout finish without introducing a scene or changing its background contract."""

    normalized = str(prompt or "").strip()
    direction = PAPER_STORYBOOK_SUBJECT_DIRECTION
    if direction in normalized:
        return normalized
    return "\n".join(part for part in (normalized, direction) if part)
