"""Character Selection responsibilities."""

from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any
from agentic.app.character_requests import (
    CharacterWorkflowRequest,
    SUBJECT_MODE_SINGLE,
    SUBJECT_MODE_INTERACTION,
    SUBJECT_MODE_RANDOM,
    CONCRETE_SUBJECT_MODES,
    SUPPORTED_SUBJECT_MODES,
)
from agentic.tools.context_services import CharacterGroupSelectionService
from agentic.app.character_config import load_character_config


def _replace_character_identity(
    text: str, *, configured_name: str, selected_name: str
) -> str:
    source = str(configured_name or "").strip()
    target = str(selected_name or "").strip()
    value = str(text or "")
    if not source or not target or source.casefold() == target.casefold():
        return value
    return re.sub(
        rf"(?<!\w){re.escape(source)}(?!\w)", target, value, flags=re.IGNORECASE
    )


def _fixed_character_selection(name: str, *, group_name: str = "") -> dict[str, Any]:
    normalized_name = str(name or "").strip()
    return {
        "mode": "config",
        "subject_mode": SUBJECT_MODE_SINGLE,
        "group_name": str(group_name or "").strip(),
        "selected_character": normalized_name,
        "candidate_count": 1,
        "candidates": [{"name": normalized_name, "status": 1, "weight": 1.0}],
        "selected_profile": {},
        "selection_source": "fixed_config",
    }


def _validate_character_path_component(name: str) -> None:
    value = str(name or "").strip()
    if not value:
        return
    invalid_path_characters = "<>:/\\|?*\x00"
    reserved_device_name = value.split(".", 1)[0].casefold() in {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    if (
        value in {".", ".."}
        or value.endswith((".", " "))
        or any(character in value for character in invalid_path_characters)
        or reserved_device_name
    ):
        raise ValueError(
            "Character name used for output paths must be a safe path component."
        )


def _subject_mode(config: dict[str, Any]) -> str:
    generation = dict(config.get("generation", {}) or {})
    mode = str(generation.get("subject_mode") or SUBJECT_MODE_SINGLE).strip().lower()
    if mode not in SUPPORTED_SUBJECT_MODES:
        raise ValueError(
            "generation.subject_mode must be 'single', 'two_character_interaction', or 'random'"
        )
    return mode


def _subject_mode_weights(config: dict[str, Any]) -> dict[str, float]:
    generation = dict(config.get("generation", {}) or {})
    raw_weights = generation.get("subject_mode_weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise ValueError(
            "generation.subject_mode_weights is required when subject_mode is 'random'"
        )

    weights: dict[str, float] = {}
    for raw_mode, raw_weight in raw_weights.items():
        mode = str(raw_mode or "").strip().lower()
        if mode not in CONCRETE_SUBJECT_MODES:
            raise ValueError(
                "generation.subject_mode_weights may contain only 'single' and "
                "'two_character_interaction'"
            )
        if mode in weights:
            raise ValueError(
                f"generation.subject_mode_weights contains duplicate mode '{mode}'"
            )
        if isinstance(raw_weight, bool):
            raise ValueError(
                f"generation.subject_mode_weights['{mode}'] must be a finite non-negative number"
            )
        try:
            weight = float(raw_weight)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"generation.subject_mode_weights['{mode}'] must be a finite non-negative number"
            ) from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"generation.subject_mode_weights['{mode}'] must be a finite non-negative number"
            )
        weights[mode] = weight

    if not any(weight > 0 for weight in weights.values()):
        raise ValueError(
            "generation.subject_mode_weights must contain at least one positive weight"
        )
    return weights


def _resolve_subject_mode(
    config: dict[str, Any],
    *,
    rng: random.Random | None,
) -> tuple[str, dict[str, float] | None]:
    configured_mode = _subject_mode(config)
    if configured_mode != SUBJECT_MODE_RANDOM:
        return configured_mode, None
    weights = _subject_mode_weights(config)
    return _weighted_choice_required(weights, rng=rng), weights


def _weighted_choice_required(
    weights: dict[str, float],
    *,
    rng: random.Random | None,
) -> str:
    chooser = rng or random
    total = sum(weights.values())
    threshold = chooser.uniform(0, total)
    cumulative = 0.0
    for mode, weight in weights.items():
        if weight <= 0:
            continue
        cumulative += weight
        if threshold <= cumulative:
            return mode
    raise RuntimeError(
        "subject mode weighted selection exhausted its validated weights"
    )


def _selection_subject_mode(
    character_selection: dict[str, Any],
    *,
    configured_subject_mode: str,
) -> str:
    if configured_subject_mode != SUBJECT_MODE_RANDOM:
        return configured_subject_mode
    selected_mode = str(character_selection.get("subject_mode") or "").strip().lower()
    if selected_mode in CONCRETE_SUBJECT_MODES:
        return selected_mode
    selection_mode = str(character_selection.get("mode") or "").strip().lower()
    if selection_mode in {"config", "group"}:
        return SUBJECT_MODE_SINGLE
    if selection_mode == SUBJECT_MODE_INTERACTION:
        return SUBJECT_MODE_INTERACTION
    raise ValueError(
        "random subject_mode requires a resolved selection with an effective subject mode"
    )


def _annotate_subject_selection(
    selection: dict[str, Any],
    *,
    configured_subject_mode: str,
    effective_subject_mode: str,
    subject_mode_weights: dict[str, float] | None,
) -> dict[str, Any]:
    resolved = dict(selection)
    _validate_character_path_component(str(resolved.get("selected_character") or ""))
    raw_subjects = resolved.get("subjects")
    if isinstance(raw_subjects, list):
        for subject in raw_subjects:
            if isinstance(subject, dict):
                _validate_character_path_component(str(subject.get("name") or ""))
    resolved["subject_mode"] = effective_subject_mode
    if configured_subject_mode == SUBJECT_MODE_RANDOM:
        resolved["configured_subject_mode"] = configured_subject_mode
        resolved["subject_mode_weights"] = dict(subject_mode_weights or {})
        resolved["subject_mode_selection_source"] = "weighted_random"
    return resolved


def _subject_context_from_selection(
    *,
    subject_mode: str,
    character_name: str,
    character_selection: dict[str, Any],
    configured_group_name: str,
) -> dict[str, Any]:
    if subject_mode == SUBJECT_MODE_INTERACTION:
        raw_subjects = character_selection.get("subjects")
        if not isinstance(raw_subjects, list) or len(raw_subjects) != 2:
            raise ValueError(
                "two_character_interaction selection must contain exactly two subject slots"
            )
        selection_is_same_group = bool(character_selection.get("is_same_group", False))
        selection_group_name = str(
            character_selection.get("group_name") or configured_group_name
        ).strip()
        subjects: list[dict[str, Any]] = []
        for index, raw_subject in enumerate(raw_subjects):
            if not isinstance(raw_subject, dict):
                raise ValueError("two_character_interaction subjects must be mappings")
            subject = dict(raw_subject)
            name = str(subject.get("name") or "").strip()
            if not name:
                raise ValueError(
                    f"two_character_interaction subject {index + 1} has no name"
                )
            subject["role"] = str(
                subject.get("role") or ("primary", "secondary")[index]
            )
            subject["name"] = name
            subject["group_name"] = str(
                subject.get("group_name") or selection_group_name
            ).strip()
            subject["profile"] = dict(subject.get("profile") or {})
            subjects.append(subject)
        return {
            "mode": subject_mode,
            "subjects": subjects,
            "selection": {
                "is_same_group": selection_is_same_group,
                "group_name": selection_group_name,
                "candidate_count": int(character_selection.get("candidate_count") or 0),
                "selection_source": str(
                    character_selection.get("selection_source") or ""
                ),
            },
            "interaction_contract": {
                "required": True,
                "same_frame": True,
                "visible_relation": True,
                "unexpected_extra_subjects": False,
            },
        }

    selected_profile = dict(character_selection.get("selected_profile") or {})
    return {
        "mode": "single",
        "subjects": [
            {
                "role": "primary",
                "name": character_name,
                "group_name": str(
                    character_selection.get("group_name") or configured_group_name
                ).strip(),
                "status": 1,
                "weight": 1.0,
                "profile": selected_profile,
            }
        ],
        "selection": {
            "is_same_group": False,
            "group_name": str(
                character_selection.get("group_name") or configured_group_name
            ).strip(),
            "candidate_count": int(character_selection.get("candidate_count") or 1),
            "selection_source": str(character_selection.get("selection_source") or ""),
        },
        "interaction_contract": {
            "required": False,
            "same_frame": False,
            "visible_relation": False,
            "unexpected_extra_subjects": True,
        },
    }


def _subject_prompt_block(subject_context: dict[str, Any]) -> str:
    subjects = list(subject_context.get("subjects") or [])
    parts: list[str] = []
    for subject in subjects:
        name = str(subject.get("name") or "").strip()
        role = str(subject.get("role") or "subject").strip()
        profile = dict(subject.get("profile") or {})
        details = "; ".join(
            item
            for item in (
                str(profile.get("role_description") or "").strip(),
                str(profile.get("keywords") or "").strip(),
            )
            if item
        )
        parts.append(f"{role}: {name}{f' ({details})' if details else ''}")
    if not parts:
        return "one main subject"
    if bool(subject_context.get("interaction_contract", {}).get("required")):
        return (
            "Two required subjects in the same readable frame: "
            + "; ".join(parts)
            + ". Show a concrete mutual interaction and preserve each identity."
        )
    return "; ".join(parts)


def resolve_character_selection(
    request: CharacterWorkflowRequest,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded_config = (
        config if config is not None else load_character_config(request.config_path)
    )
    character = dict(loaded_config.get("character", {}) or {})
    configured_name = str(character.get("name") or Path(request.config_path).stem)
    group_name = str(character.get("group_name") or "").strip()
    requested_subject_mode = str(request.generation.subject_mode or "").strip().lower()
    if requested_subject_mode and requested_subject_mode not in SUPPORTED_SUBJECT_MODES:
        raise ValueError(f"Unsupported subject mode: {requested_subject_mode}")
    configured_subject_mode = requested_subject_mode or _subject_mode(loaded_config)
    requested_character_name = str(
        request.generation.selected_character_name or ""
    ).strip()
    story_card_requested = (
        str(request.generation.preferred_generation_type or "").strip().lower()
        == "story_card"
    )
    reference_video_requested = str(
        request.generation.preferred_generation_type or ""
    ).strip().lower() == "text2image2video" and bool(
        str(request.generation.reference_video_source or "").strip()
    )
    game_sprite_requested = (
        str(request.generation.preferred_generation_type or "").strip().lower()
        == "game_sprite"
    )
    if story_card_requested:
        # Story cards are writing-first decorative Kirby cards. Random or
        # two-character selection adds unrelated subjects and can change the
        # character identity that the corner illustration is meant to keep.
        if group_name:
            selection = (
                CharacterGroupSelectionService()
                .select_named_character(
                    group_name,
                    configured_name,
                )
                .to_dict()
            )
        else:
            selection = _fixed_character_selection(configured_name)
        return _annotate_subject_selection(
            selection,
            configured_subject_mode=configured_subject_mode,
            effective_subject_mode=SUBJECT_MODE_SINGLE,
            subject_mode_weights=None,
        )
    if requested_character_name and configured_subject_mode == SUBJECT_MODE_SINGLE:
        selection = (
            CharacterGroupSelectionService()
            .select_named_character(
                group_name,
                requested_character_name,
            )
            .to_dict()
            if group_name
            else _fixed_character_selection(requested_character_name)
        )
        return _annotate_subject_selection(
            selection,
            configured_subject_mode=configured_subject_mode,
            effective_subject_mode=SUBJECT_MODE_SINGLE,
            subject_mode_weights=None,
        )
    if reference_video_requested or game_sprite_requested:
        # Reference-driven I2V and game sprites use one readable protagonist.
        # Do not let the general Kirby config's interaction weighting add an
        # unrelated second subject to these technical routes.
        configured_subject_mode = SUBJECT_MODE_SINGLE
    if reference_video_requested or game_sprite_requested:
        subject_mode, subject_mode_weights = SUBJECT_MODE_SINGLE, None
    else:
        selection_config = loaded_config
        if requested_subject_mode:
            selection_config = dict(loaded_config)
            selection_generation = dict(loaded_config.get("generation", {}) or {})
            selection_generation["subject_mode"] = requested_subject_mode
            selection_config["generation"] = selection_generation
        subject_mode, subject_mode_weights = _resolve_subject_mode(
            selection_config,
            rng=request.generation.rng,
        )
    if subject_mode == SUBJECT_MODE_INTERACTION:
        generation = dict(loaded_config.get("generation", {}) or {})
        interaction = dict(generation.get("two_character_interaction", {}) or {})
        is_same_group = bool(interaction.get("is_same_group", True))
        interaction_group_name = str(
            interaction.get("group_name") or group_name
        ).strip()
        selection = (
            CharacterGroupSelectionService()
            .select_pair(
                interaction_group_name or None,
                is_same_group=is_same_group,
                rng=request.generation.rng,
            )
            .to_dict()
        )
        return _annotate_subject_selection(
            selection,
            configured_subject_mode=configured_subject_mode,
            effective_subject_mode=subject_mode,
            subject_mode_weights=subject_mode_weights,
        )
    if not group_name:
        selection = _fixed_character_selection(configured_name)
        return _annotate_subject_selection(
            selection,
            configured_subject_mode=configured_subject_mode,
            effective_subject_mode=subject_mode,
            subject_mode_weights=subject_mode_weights,
        )
    selection = CharacterGroupSelectionService().select_random_character(
        group_name,
        rng=request.generation.rng,
    )
    selection_payload = selection.to_dict()
    prompt_text = str(request.generation.prompt or "").casefold()
    if prompt_text and isinstance(selection_payload.get("candidates"), list):
        named_candidate = next(
            (
                candidate
                for candidate in sorted(
                    selection_payload["candidates"],
                    key=lambda item: (
                        len(str(item.get("name") or ""))
                        if isinstance(item, dict)
                        else 0
                    ),
                    reverse=True,
                )
                if isinstance(candidate, dict)
                and str(candidate.get("name") or "").strip().casefold() in prompt_text
            ),
            None,
        )
        if named_candidate is not None:
            selection_payload["selected_character"] = str(named_candidate["name"])
            selection_payload["selected_profile"] = {
                "role_description": str(named_candidate.get("role_description") or ""),
                "keywords": str(named_candidate.get("keywords") or ""),
            }
            selection_payload["selection_source"] = "prompt_named_role"
    return _annotate_subject_selection(
        selection_payload,
        configured_subject_mode=configured_subject_mode,
        effective_subject_mode=subject_mode,
        subject_mode_weights=subject_mode_weights,
    )
