"""Readable episode, scene and shot planning for serial video."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

from agentic.runtime.llm_client import PromptClient


SERIAL_EPISODE_PLAN_SYSTEM_PROMPT = """你是動畫短劇編劇。已核准的系列故事與角色卡是依據；本集方向只補充重點，不可更換主角或推翻設定。

用繁體中文寫出容易理解、適合短動畫的本集故事與畫面安排，不輸出 JSON、資料表或程式碼。讓每個動作都清楚可見，劇情有開始、問題和結果；依創作者選擇的複雜度決定細節多寡。"""


def build_serial_episode_outline_prompts(
    *,
    approved_story: str,
    approved_character: str,
    episode_number: int,
    episode_topic: str,
    episode_direction: str = "",
    previous_episode_summaries: Sequence[str] = (),
    complexity: str = "simple",
) -> tuple[str, str]:
    story = str(approved_story or "").strip()
    character = str(approved_character or "").strip()
    topic = str(episode_topic or "").strip()
    number = int(episode_number)
    normalized_complexity = str(complexity or "").strip().lower()
    if normalized_complexity not in {"simple", "cinematic"}:
        raise ValueError("Serial story complexity must be simple or cinematic.")
    if not story or not character:
        raise ValueError("Episode planning requires an approved story and character card.")
    if number < 1:
        raise ValueError("Episode number must be positive.")
    if not topic:
        raise ValueError("Episode planning requires a human-readable episode topic.")
    previous = [str(item).strip() for item in previous_episode_summaries if str(item).strip()]
    prompt_parts = [
        "以下是已核准的完整系列故事，必須保留它的主線、角色改變與結局方向：",
        story,
        "",
        "以下是已核准的主角身分卡，維持角色的外觀、性格與行為：",
        character,
        f"\n本次規劃第 {number} 集，集名／主題：{topic}",
        "本集創作者方向：",
        str(episode_direction or "").strip() or "（沒有額外方向，請依系列故事自然推進本集。）",
    ]
    if normalized_complexity == "simple":
        prompt_parts.extend(
            [
                "",
                "本集複雜度：simple。只講一件小事：主角的一個目標、一個看得見的麻煩、一個簡單解法和清楚結果。最多兩個場景，不加支線、額外角色或反派。用三個節拍寫：開始、遇到問題、解決問題。每個節拍只安排一個主要動作，使用日常詞彙；不寫大場面、長篇解說或反轉。",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "本集複雜度：cinematic。可以安排較多因果轉折、情感變化或場景，但每一個轉折都要服務同一條主線，並讓觀眾能理解角色為何這樣做。",
            ]
        )
    if previous:
        prompt_parts.extend(
            [
                "",
                "已核准的前集結尾摘要，必須從最新狀態接續：",
                *[f"- {summary}" for summary in previous],
            ]
        )
    if normalized_complexity == "simple":
        prompt_parts.extend(
            [
                "",
                "請只輸出三部分：本集一句話；開始、問題、解決三個短節拍；最多五個簡單畫面提示，交代角色做什麼和畫面如何結束即可。省略全系列分析、分鏡術語、鏡頭運動和道具狀態表。",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "請依照以下順序完成本集規劃：",
                "一、本集在全系列中的目的，以及它如何承接上集並推進主線。",
                "二、按因果順序列出本集劇情節點：開場狀態、目標、阻礙如何升級、角色作出的選擇、高潮轉折、結尾狀態。",
                "三、依劇情節點拆成場景，再把場景拆成適合連續影片生成的短鏡頭。",
                "四、每個鏡頭依序列出目的、開場狀態、角色動作節拍、道具變化、構圖或鏡頭移動、結尾狀態。描述要具體到能交給影片生成流程，並指出必要時人物、位置、道具或情緒如何延續。",
                "故事內容要完整推進本集的事件，鏡頭清單不可取代劇情；不確定集數時以最自然的篇幅規劃，不為增加集數重複事件。",
            ]
        )
    return SERIAL_EPISODE_PLAN_SYSTEM_PROMPT, "\n".join(prompt_parts)


def generate_serial_episode_outline(
    client: PromptClient,
    *,
    approved_story: str,
    approved_character: str,
    episode_number: int,
    episode_topic: str,
    episode_direction: str = "",
    previous_episode_summaries: Sequence[str] = (),
    complexity: str = "simple",
) -> str:
    system_prompt, user_prompt = build_serial_episode_outline_prompts(
        approved_story=approved_story,
        approved_character=approved_character,
        episode_number=episode_number,
        episode_topic=episode_topic,
        episode_direction=episode_direction,
        previous_episode_summaries=previous_episode_summaries,
        complexity=complexity,
    )
    return client.generate_text(
        system_prompt,
        user_prompt,
        step_name="serial_episode_story_scenes_shots",
        language_contract="請使用自然、易讀的繁體中文。",
    )


def serial_episode_outline_artifact_path(
    output_root: Path,
    series_id: str,
    episode_number: int,
    outline_id: str,
) -> Path:
    raw_series_id = str(series_id or "").strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_series_id).strip(".-_") or "series"
    slug = slug[:48]
    identity = hashlib.sha256(raw_series_id.encode("utf-8")).hexdigest()[:10]
    outline = re.sub(r"[^A-Fa-f0-9]", "", str(outline_id or ""))
    if len(outline) != 32:
        raise ValueError("An episode outline ID must contain 32 hexadecimal characters.")
    number = int(episode_number)
    if number < 1:
        raise ValueError("Episode number must be positive.")
    return (
        Path(output_root)
        / "serial_series"
        / f"{slug}-{identity}"
        / "episodes"
        / f"episode-{number}-{outline}.md"
    )
