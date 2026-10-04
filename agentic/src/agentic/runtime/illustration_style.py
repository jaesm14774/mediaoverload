from __future__ import annotations


PAPER_STORYBOOK_ART_DIRECTION = (
    "Default illustration art direction: hand-painted watercolor and gouache with tactile pigment, confident "
    "ink contours, dimensional shadows, and a specific scene environment that grounds the characters' action. "
    "Keep the characters and their physical interaction dominant; use a readable foreground, middle distance, "
    "and background instead of blank paper. Match composition, color, face, and body posture to the emotion. "
    "For comedy, use vivid contrast, a strong diagonal silhouette, exaggerated body acting, clear physical "
    "contact, and open laughing or startled expressions. For tenderness or sadness, bring the characters "
    "closer and let posture and light carry the feeling without forcing a smile. Preserve the requested subject "
    "and honor an explicitly requested medium, palette, or isolated background."
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
