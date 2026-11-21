"""Readable episode, scene and shot planning for serial video."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agentic.runtime.serial_ledger import SERIES_MAX_EPISODES, is_final_episode


SERIAL_EPISODE_PLAN_SYSTEM_PROMPT = """你是動畫短劇編劇。已核准的系列故事與角色卡是依據；本集方向只補充重點，不可更換主角或推翻設定。

用繁體中文寫出容易理解、適合短動畫的本集故事與畫面安排，不輸出 JSON、資料表或程式碼。讓每個動作都清楚可見，劇情有開始、問題和結果；依創作者選擇的複雜度決定細節多寡。"""


LEDGER_HEADER = "線索帳本的格式（程式會逐行檢查，只能寫以下條列，不可寫其他文字）："
LEDGER_ADD_MAIN = "- 新增 主線｜（只有第一集需要）觀眾想知道答案的長期問題，系列結束前不會被回答"
LEDGER_ADD = "- 新增｜本集埋下的新線索，一行一條，最多一兩條"
LEDGER_ADVANCE = "- 推進 T編號｜這條開放中的線索在本集往前走了什麼（要引用上面列出的編號，例如 T1）"
LEDGER_PAY = "- 回收 T編號｜本集揭曉的線索（一集最多一條，主線不可回收）"
LEDGER_PAY_ALL = "- 回收 T編號｜每一條開放中的線索各寫一行，包含主線；最終集必須全部回收，不能再新增線索"
LEDGER_PRESENT = "- 出場｜第一個節拍就在畫面上的常駐角色名字（不含主角），用「、」分隔；沒有常駐角色出場就不要寫這一行"
LEDGER_NEXT_GOAL = "- 下集目標｜這一集結尾留給下一集的未完成目標（必須剛好一行）"


def ledger_format(episode_number: int) -> str:
    """Ledger section format; ids are assigned by the program, so only earlier episodes' threads can be cited."""
    if episode_number == 1:
        lines = [
            LEDGER_HEADER,
            LEDGER_ADD_MAIN,
            LEDGER_ADD,
            LEDGER_PRESENT,
            LEDGER_NEXT_GOAL,
            "第一集還沒有任何線索編號，不要寫「推進」或「回收」；新增的線索由程式在本集被接受後編號。",
        ]
    elif is_final_episode(episode_number):
        lines = [
            LEDGER_HEADER,
            LEDGER_PAY_ALL,
            LEDGER_PRESENT,
            "- 下集目標｜寫「無：全劇完結」（必須剛好一行）",
            "只能引用上面列出的編號。",
        ]
    else:
        lines = [
            LEDGER_HEADER,
            LEDGER_ADD_MAIN,
            LEDGER_ADD,
            LEDGER_ADVANCE,
            LEDGER_PAY,
            LEDGER_PRESENT,
            LEDGER_NEXT_GOAL,
            "本集新增的線索還沒有編號，不能在本集推進或回收；只能引用上面列出的編號。",
        ]
    return "\n".join(lines)


def build_serial_episode_outline_prompts(
    *,
    approved_story: str,
    approved_character: str,
    episode_number: int,
    episode_topic: str,
    episode_direction: str = "",
    approved_story_state: str = "",
    series_recap: str = "",
    ledger_text: str = "",
    roster_text: str = "",
    beat_count: int = 4,
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
    if int(beat_count) < 1:
        raise ValueError("An episode needs at least one beat.")
    beats = int(beat_count)
    recap = str(series_recap or "").strip()
    ledger = str(ledger_text or "").strip()
    roster = str(roster_text or "").strip()
    current_state = str(approved_story_state or "").strip()
    final = is_final_episode(number)
    prompt_parts = [
        "以下是已核准的完整系列故事，必須保留它的主線與角色設定：",
        story,
        "",
        "以下是已核准的主角身分卡，維持角色的外觀、性格與行為：",
        character,
        f"\n本次規劃第 {number} 集，集名／主題：{topic}",
        "本集創作者方向：",
        str(episode_direction or "").strip() or "（沒有額外方向，請依系列故事自然推進本集。）",
        "",
        (
            f"本集是全劇最後一集（全劇共 {SERIES_MAX_EPISODES} 集，總長不到五分鐘）：把線索帳本裡開放中的主線和所有線索都在本集演完，給主角一個明確、讓人滿足的結果。"
            if final
            else f"只演第 {number} 集（全劇共 {SERIES_MAX_EPISODES} 集，總長不到五分鐘）：已核准故事中的路線只當背景；本集只演它自己的部分，之後的集數是未來，不得提前演出，也不得完成長期懸念或主角的最終目標。"
        ),
    ]
    if normalized_complexity == "simple":
        prompt_parts.extend(
            [
                "",
                f"本集複雜度：simple。這是約 {beats * 10} 秒的微短劇一集，短到只裝得下：一個小目標、一個看得見的麻煩、一個簡單解法和小結果，只有一個場景，不加反派或支線。每個節拍只安排一個目標，用 2 到 3 個連續的大動作完成，其中至少一個是大位移（奔跑、跳、飛、翻滾、被拋出、追逐或推著東西衝過畫面），只在原地整理、接球、遞東西或捲紙不算，使用日常詞彙；不寫大場面或長篇解說，但可以有一個小反轉：中途讓一個預期落空，再用角色自己的選擇接住它。劇情只用鏡頭拍得到的動作和大件、顏色鮮明、一眼認得的道具（球、鑰匙、果凍、帽子、氣球）來講：不要靠文字、字條、海報、招牌、手機或螢幕訊息、對白或聲音傳達劇情，也不使用真實的國家、地名、機構或人物。每個節拍會用上一段的最後畫面接著生成：第一個節拍就要讓之後用到的所有地點、道具和常駐角色入鏡，後面的節拍不能憑空換景或冒出新東西；物品的位置前後一致（東西掉進哪裡，下一個節拍就在那裡找）；有常駐角色出場就在每個節拍都留在畫面裡，並做出自己的動作。",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "本集複雜度：cinematic。可以安排較多因果轉折、情感變化或場景，但每一個轉折都要服務同一條主線，並讓觀眾能理解角色為何這樣做。",
            ]
        )
    prompt_parts.extend(
        [
            "",
            (
                "結局規則：這是最終集，結尾是主線被解開後的明確收束畫面，不留鉤子，也不留下一集的目標。"
                if final
                else "連續劇規則：結尾必須留下鉤子，也就是一個新的、尚未回答的小問題或意外畫面，讓人想看下一集；本集不得解開長期懸念。本集仍要有自己的小結果，不能只丟問題。"
            ),
        ]
    )
    if recap:
        prompt_parts.extend(
            [
                "",
                "已播出的集數與每集交給下一集的目標（本集必須從最後一集交出的目標接續，不得重演已演過的差事、場景或動作）：",
                recap,
            ]
        )
    if ledger and number > 1:
        prompt_parts.extend(["", "目前系列還欠觀眾的線索（以編號引用）：", ledger])
    if roster:
        prompt_parts.extend(
            [
                "",
                "本集的常駐角色班底（每個常駐角色都要在第一個節拍入鏡，並且每個節拍都有自己推進故事的動作，與主角互動，不是站在旁邊看）：",
                roster,
            ]
        )
    if number > 1:
        prompt_parts.extend(
            [
                "",
                "因果規則：本集一句話必須用「因此」或「但是」連到上一集交出的目標；不可用「同時」「另一邊」硬接。至少推進一條開放中的線索（用線索編號），並讓本集結束時有一個看得見的改變：關係、知情內容、持有物或地點其中之一。",
            ]
        )
    else:
        prompt_parts.extend(
            ["", "這是系列的第一集：用本集埋下觀眾想知道答案的長期問題（寫成「新增 主線｜…」），並埋下一兩個之後可以回收的小線索。"]
        )
    if current_state:
        prompt_parts.extend(
            [
                "",
                "這是上一集經創作者接受後保存的正式連續狀態；它描述已核准發生的事，優先於模型記憶或草稿摘要：",
                current_state,
                "新一集必須從此狀態出發；除非本集故事明確造成改變，否則不得重置人物位置、身體／情緒狀態、角色知情內容、關係、線索、道具持有人／位置／狀況或場景變化。",
            ]
        )
    if normalized_complexity == "simple":
        ending = "解決主線並收尾" if final else "解決並留下鉤子"
        beat_names = {
            3: f"開場帶出麻煩、轉折、{ending}",
            4: f"開場、問題、升級、{ending}",
        }.get(beats, f"依序推進的 {beats} 個節拍，最後一個節拍{'收尾' if final else '以鉤子收尾'}")
        next_goal = "寫「無：全劇完結」" if final else "留給下一集的未完成目標（也就是鉤子）"
        prompt_parts.extend(
            [
                "",
                f"請輸出四部分：一、本集一句話，只寫看得見的事：主角想要什麼、遇到什麼麻煩、怎麼收尾（長期問題只寫在線索帳本，不要寫進這一句，也不要寫進節拍）；二、{_chinese_number(beats)}個節拍（{beat_names}），每個節拍對應一支 10 秒影片，每個節拍只寫一到兩句、約 40 字，只寫該段看得見的一個目標，以及完成它的 2 到 3 個連續、大幅度的肢體動作與因果；每個節拍（包含開場）都從第一秒就開始動，睡覺、發呆、呼吸、發光、靜靜漂浮、站著看都不算動作；每個節拍結束時，主角必須仍在畫面中央、清楚可見，每個常駐角色也必須完整留在畫面內、分站主角左右、看得見臉，不被畫面邊緣切到，不得離開畫面、飄走或退到背景，因為下一支影片只能接續上一支的最後一格，最後一格裡看不清楚的角色之後無法再畫回來；結束時所有角色仍在做動作（跳、揮手、轉圈、追逐、歡呼），不可站定、停步或擺姿勢，結尾狀態寫動作中的位置，不寫「站定」「面向鏡頭」；三、本集結尾狀態卡，逐項寫人物位置／身體與情緒狀態／新知道的事、關係變化、線索狀態、重要道具持有人／位置／狀況、場景變化，以及{next_goal}；四、線索帳本（格式見下）。狀態卡要描述本集結尾後會實際延續的狀態，沒有變化的項目寫「不變」，沒有相關內容寫「無」。不用輸出 JSON、分鏡術語或鏡頭運動。",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "請依照以下順序完成本集規劃：",
                "一、本集在全系列中的目的，以及它如何承接上集並推進主線。",
                "二、按因果順序列出本集劇情節點：開場狀態、目標、阻礙如何升級、角色作出的選擇、高潮轉折、結尾狀態。",
                f"三、依劇情節點拆成場景，再把場景拆成恰好 {beats} 個鏡頭，每個鏡頭對應一支影片（10 秒），劇情節點要依序分配到這 {beats} 個鏡頭裡，最後一個鏡頭收在本集的小結果與鉤子；不可多於或少於 {beats} 個鏡頭。",
                "四、每個鏡頭依序列出目的、開場狀態、角色動作節拍、道具變化、構圖、結尾狀態。鏡頭固定，不要推近、旋轉、搖鏡、特寫切換或升降；變化全由角色和場景的動作帶來。每個鏡頭的角色動作節拍都是連續、大幅度的肢體動作，第一秒就開始動，且至少一個是大位移（奔跑、跳、飛、翻滾、被拋出、追逐或推著東西衝過畫面），只在原地整理、接球、遞東西或捲紙不算；睡覺、發呆、呼吸、發光、靜靜漂浮都不算動作。每個鏡頭的結尾狀態，主角必須仍在畫面中央、清楚可見，每個常駐角色也必須完整留在畫面內、分站主角左右、看得見臉，不被畫面邊緣切到，不得離開畫面、飄走或退到背景，因為下一支影片只能接續上一支的最後一格，最後一格裡看不清楚的角色之後無法再畫回來；結束時所有角色仍在做動作（跳、揮手、轉圈、追逐、歡呼），不可站定、停步或擺姿勢，結尾狀態寫動作中的位置，不寫「站定」「面向鏡頭」。描述要具體到能交給影片生成流程，並指出必要時人物、位置、道具或情緒如何延續。",
                "五、本集結尾狀態卡：寫出人物位置／身體與情緒狀態／新知道的事、關係變化、線索狀態、重要道具持有人／位置／狀況、場景變化，以及留給下一集的未完成目標。沒有變化寫「不變」，沒有相關內容寫「無」。",
                "六、線索帳本（格式見下）。",
                "故事內容要完整推進本集的事件，鏡頭清單不可取代劇情；不確定集數時以最自然的篇幅規劃，不為增加集數重複事件。",
            ]
        )
    prompt_parts.extend(["", ledger_format(number)])
    return SERIAL_EPISODE_PLAN_SYSTEM_PROMPT, "\n".join(prompt_parts)


def episode_summary(outline_text: str) -> str:
    """The outline's opening one-liner: what the episode showed, for later recaps."""
    first_section = re.split(r"\n\s*二、", str(outline_text or ""), maxsplit=1)[0]
    lines = [line.strip() for line in first_section.splitlines() if line.strip()]
    return " ".join(lines[1:] if len(lines) > 1 else lines)


def build_ledger_repair_prompt(
    *,
    outline_text: str,
    episode_number: int,
    ledger_text: str,
    roster_text: str,
    error: str,
) -> str:
    """Ask for the ledger section alone, so a rejected ledger never costs the story that was already written."""
    parts = [
        "以下是已經寫好的本集大綱，故事部分已定案，不要改：",
        str(outline_text).strip(),
        "",
        f"它的線索帳本被檢查退回：{error}",
    ]
    if int(episode_number) > 1 and str(ledger_text or "").strip():
        parts.extend(["", "目前系列還欠觀眾的線索（以編號引用）：", str(ledger_text).strip()])
    if str(roster_text or "").strip():
        parts.extend(["", "可以回歸的常駐角色（出場只能寫這些名字，一字不差）：", str(roster_text).strip()])
    parts.extend(
        [
            "",
            "請依上面的故事重寫線索帳本，只輸出線索帳本這一段（標題寫「線索帳本」），不要輸出大綱的其他部分。",
            "",
            ledger_format(int(episode_number)),
        ]
    )
    return "\n".join(parts)


_BOOKKEEPING_HEADING = re.compile(r"線索帳本|結尾狀態卡|本集結束後的連續狀態")
_SECTION_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+|(?:\*\*)?(?:[一二三四五六七八九十]+[、.．]|線索帳本|本集結尾狀態卡|結尾狀態卡|本集結束後的連續狀態))"
)
# Ledger entries are written with the bar the format prescribes; a story line such as
# 「出場：Dede 從樹後跳出來」 is story, not bookkeeping.
_LEDGER_ENTRY = re.compile(r"^[ \t]*[-*•]?[ \t]*(?:新增|推進|回收|下集目標|出場)(?:[ \t]+(?:主線|T\d+))?[ \t]*[｜|]")


def episode_story_text(outline_text: str) -> str:
    """The outline without the end-state card and thread ledger; those are for review, not for the video planner."""
    kept: list[str] = []
    skipping = False
    for line in str(outline_text or "").splitlines():
        if _SECTION_HEADING.match(line):
            skipping = _BOOKKEEPING_HEADING.search(line) is not None
        if skipping or _LEDGER_ENTRY.match(line):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _chinese_number(value: int) -> str:
    return {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五", 6: "六"}.get(value, str(value))


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
