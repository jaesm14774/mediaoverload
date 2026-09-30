from __future__ import annotations


PAPER_STORYBOOK_ART_DIRECTION = (
    "Default illustration art direction: a refined, atmospheric storybook image on warm ivory paper "
    "with subtle handmade fiber and pigment texture; hand-painted watercolor and gouache with delicate "
    "ink or pencil edges; layered foreground, middle distance, and background; nuanced natural light and "
    "soft shadows; a restrained yet clearly chromatic palette with considered cool blues, warm neutrals, "
    "and muted coral or ochre accents. Build one clear focal path and let generous negative space retain "
    "faint scene texture, light, shadow, or environmental traces so it feels intentional rather than blank. "
    "Keep fine material detail and an original, quietly poetic mood. Preserve the requested subject and "
    "honor an explicitly requested different medium, palette, or isolated background."
)
PAPER_STORYBOOK_SUBJECT_DIRECTION = (
    "Default subject art direction: render the subject itself as a refined hand-painted storybook "
    "illustration with watercolor and gouache pigment, delicate ink or pencil edges, subtle paper-like "
    "surface detail within the forms, nuanced light and soft dimensional shadows, and a restrained yet "
    "clearly chromatic palette. Preserve the requested silhouette and any transparent, solid, or chroma-key "
    "background exactly; do not add a floor or environment when the asset requires isolation. Honor an "
    "explicitly requested different medium or palette."
)


def apply_paper_storybook_art_direction(prompt: str, *, isolated_subject: bool = False) -> str:
    """Add the shared visual direction while preserving cutout background contracts."""

    normalized = str(prompt or "").strip()
    direction = PAPER_STORYBOOK_SUBJECT_DIRECTION if isolated_subject else PAPER_STORYBOOK_ART_DIRECTION
    if direction in normalized:
        return normalized
    return "\n".join(part for part in (normalized, direction) if part)
