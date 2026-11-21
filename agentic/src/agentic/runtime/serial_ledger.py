"""Thread ledger for a continuing series: what viewers were promised and what is still open.

An approved episode outline carries a strict ``線索帳本`` section. It is parsed and checked
here with plain rules (no model judges the story); the creator still decides in Discord.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

# A thread nobody touched for this many episodes must be touched by the next one.
DUE_AFTER_EPISODES = 3
MAX_OPEN_THREADS = 6
MAX_PAYOFFS_PER_EPISODE = 1
# A micro-drama: episodes run 30-40 s, so seven of them stay under five minutes.
MAX_EPISODE_SECONDS = 40
# One H3 clip carries one beat, so an episode is a whole number of clips.
SERIAL_CLIP_SECONDS = 10
SERIES_MAX_EPISODES = 7


def validate_episode_seconds(duration_seconds: int) -> None:
    """An episode is 1-4 whole ten-second clips, so the whole series stays under five minutes."""
    seconds = int(duration_seconds)
    if seconds < SERIAL_CLIP_SECONDS or seconds > MAX_EPISODE_SECONDS or seconds % SERIAL_CLIP_SECONDS:
        raise ValueError(
            f"A serial episode runs {SERIAL_CLIP_SECONDS}, 20, 30 or {MAX_EPISODE_SECONDS} seconds "
            f"(whole {SERIAL_CLIP_SECONDS} second clips, at most {MAX_EPISODE_SECONDS}) so the whole series stays under five minutes; got {seconds}"
        )


def is_final_episode(episode_number: int) -> bool:
    """The last episode the series may have: it closes the main thread instead of hooking a next one."""
    return int(episode_number) >= SERIES_MAX_EPISODES


class LedgerError(ValueError):
    """The outline's ledger section is missing, malformed, or breaks a continuity rule."""


class PriorEpisodeNotAccepted(LedgerError):
    """The previous episode has no accepted ledger yet; rewriting this outline cannot fix that."""


@dataclass(frozen=True)
class Thread:
    id: str
    text: str
    main: bool
    planted: int
    touched: int
    paid: int | None = None


@dataclass(frozen=True)
class EpisodeDelta:
    """Ledger changes one episode makes. New threads get their ids when committed."""

    adds: tuple[tuple[str, bool], ...]
    advances: tuple[tuple[str, str], ...]
    pays: tuple[tuple[str, str], ...]
    next_goal: str
    present: tuple[str, ...]


_SECTION = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?[ \t]*(?:[一二三四五六七八九十]+[、.．][ \t]*)?線索帳本[^\n]*$"
)
_NEXT_SECTION = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]+|(?:\*\*)?[一二三四五六七八九十]+[、.．])"
)
_NOTHING = frozenset({"無", "沒有", "none", "n/a"})
_RULE = re.compile(r"^[-*_]{3,}$")
_ENTRY = re.compile(
    r"^[-*•][ \t]*(新增|推進|回收|下集目標|出場)(?:[ \t]+(主線|T\d+))?[ \t]*[｜|：:][ \t]*(.+)$"
)


def replace_ledger_section(outline_text: str, ledger_section: str) -> str:
    """The outline with its ledger section replaced (or added), every other section untouched."""
    text = str(outline_text or "")
    heading = _SECTION.search(text)
    if heading is not None:
        following = _NEXT_SECTION.search(text[heading.end():])
        end = heading.end() + following.start() if following is not None else len(text)
        text = text[: heading.start()] + text[end:]
    section = str(ledger_section or "").strip()
    if _SECTION.search(section) is None:
        section = f"線索帳本\n{section}"
    return f"{text.rstrip()}\n\n{section}\n"


def parse_episode_delta(outline_text: str, *, protagonist: str = "") -> EpisodeDelta:
    """Read the strict ledger section of an episode outline.

    The protagonist is always on screen, so naming them in 出場 is harmless and dropped.
    """
    text = str(outline_text or "")
    heading = _SECTION.search(text)
    if heading is None:
        raise LedgerError("大綱缺少「線索帳本」段落；每集都必須列出新增、推進、回收的線索和下集目標。")
    body = text[heading.end():]
    following = _NEXT_SECTION.search(body)
    if following is not None:
        body = body[: following.start()]

    adds: list[tuple[str, bool]] = []
    advances: list[tuple[str, str]] = []
    pays: list[tuple[str, str]] = []
    goals: list[str] = []
    present: list[str] = []
    for raw in body.splitlines():
        line = raw.replace("**", "").strip()
        if not line or _RULE.match(line):
            continue
        entry = _ENTRY.match(line)
        if entry is None:
            raise LedgerError(f"線索帳本只能寫條列項目，這一行不符合格式：{line}")
        kind, modifier, value = entry.group(1), entry.group(2), entry.group(3).strip()
        if kind != "下集目標" and value.strip("（）() 。").casefold() in _NOTHING:
            continue
        if kind == "新增":
            if modifier not in (None, "主線"):
                raise LedgerError(f"新增不能指定線索編號：{line}")
            adds.append((value, modifier == "主線"))
        elif kind in ("推進", "回收"):
            if not modifier or not modifier.startswith("T"):
                raise LedgerError(f"{kind}必須指定線索編號，例如「{kind} T1｜…」：{line}")
            (advances if kind == "推進" else pays).append((modifier, value))
        elif kind == "下集目標":
            if modifier:
                raise LedgerError(f"下集目標不能帶編號：{line}")
            goals.append(value)
        else:
            if modifier:
                raise LedgerError(f"出場不能帶編號：{line}")
            present.extend(
                name.strip()
                for name in re.split(r"[、,，]", value)
                if name.strip() and name.strip().casefold() != protagonist.strip().casefold()
            )
    if len(goals) != 1:
        raise LedgerError("線索帳本必須剛好有一行「下集目標｜…」，作為這集交給下一集的未完成目標。")
    return EpisodeDelta(
        adds=tuple(adds),
        advances=tuple(advances),
        pays=tuple(pays),
        next_goal=goals[0],
        present=tuple(dict.fromkeys(present)),
    )


def _open(ledger: Sequence[Thread]) -> list[Thread]:
    return [thread for thread in ledger if thread.paid is None]


def due_threads(ledger: Sequence[Thread], next_episode_number: int) -> list[Thread]:
    return [
        thread
        for thread in _open(ledger)
        if next_episode_number - thread.touched > DUE_AFTER_EPISODES
    ]


def validate_episode_delta(
    delta: EpisodeDelta,
    ledger: Sequence[Thread],
    *,
    episode_number: int,
    roster_names: Iterable[str],
) -> None:
    """Raise LedgerError when the delta breaks a continuity rule against the current ledger."""
    final = is_final_episode(episode_number)
    if int(episode_number) > SERIES_MAX_EPISODES:
        raise LedgerError(f"全劇最多 {SERIES_MAX_EPISODES} 集（總長不超過五分鐘），第 {SERIES_MAX_EPISODES} 集已是最終集。")
    open_threads = _open(ledger)
    open_ids = {thread.id: thread for thread in open_threads}
    listed = "、".join(open_ids) or "（目前沒有開放中的線索）"

    cited = [thread_id for thread_id, _ in (*delta.advances, *delta.pays)]
    for thread_id in cited:
        if thread_id not in open_ids:
            raise LedgerError(f"線索 {thread_id} 不存在或已回收；開放中的線索：{listed}。")
    if len(set(cited)) != len(cited):
        raise LedgerError("同一條線索在一集內只能推進或回收一次。")
    for thread_id, _ in delta.pays:
        if open_ids[thread_id].main and not final:
            raise LedgerError(
                f"{thread_id} 是主線，只有最終集（第 {SERIES_MAX_EPISODES} 集）才能回收；之前的集數只能推進它。"
            )
    if final:
        if delta.adds:
            raise LedgerError("最終集不能再新增線索；請把開放中的線索收掉。")
        unpaid = [thread.id for thread in open_threads if thread.id not in {i for i, _ in delta.pays}]
        if unpaid:
            raise LedgerError(
                f"最終集必須回收所有開放中的線索（含主線），還沒回收：{'、'.join(unpaid)}。"
            )
    elif len(delta.pays) > MAX_PAYOFFS_PER_EPISODE:
        raise LedgerError(f"一集最多回收 {MAX_PAYOFFS_PER_EPISODE} 條線索。")

    if open_threads and not cited:
        raise LedgerError(f"本集沒有推進或回收任何開放中的線索（{listed}）；至少要推進一條。")
    due = due_threads(ledger, episode_number)
    if due and not any(thread.id in cited for thread in due):
        due_ids = "、".join(thread.id for thread in due)
        raise LedgerError(f"線索 {due_ids} 已超過 {DUE_AFTER_EPISODES} 集沒被碰到；本集至少要推進或回收其中一條。")

    after = apply_delta(ledger, delta, episode_number=episode_number)
    open_after = _open(after)
    if not final and not any(thread.main for thread in open_after):
        raise LedgerError("系列必須一直有一條開放中的主線；請用「新增 主線｜…」寫出觀眾想知道答案的長期問題。")
    if len(open_after) > MAX_OPEN_THREADS:
        raise LedgerError(f"開放中的線索最多 {MAX_OPEN_THREADS} 條；請少新增，或回收一條舊的。")

    roster = tuple(roster_names)
    for name in delta.present:
        if name not in roster:
            known = "、".join(roster) or "（沒有常駐角色）"
            raise LedgerError(f"出場角色「{name}」不在常駐角色名單中；名單：{known}。")


def apply_delta(
    ledger: Sequence[Thread], delta: EpisodeDelta, *, episode_number: int
) -> tuple[Thread, ...]:
    """Return the ledger after the episode; new threads are numbered after the existing ones."""
    threads = {thread.id: thread for thread in ledger}
    for thread_id, _ in delta.advances:
        threads[thread_id] = replace(threads[thread_id], touched=episode_number)
    for thread_id, _ in delta.pays:
        threads[thread_id] = replace(
            threads[thread_id], touched=episode_number, paid=episode_number
        )
    next_number = len(threads) + 1
    for text, main in delta.adds:
        thread_id = f"T{next_number}"
        threads[thread_id] = Thread(
            id=thread_id, text=text, main=main, planted=episode_number, touched=episode_number
        )
        next_number += 1
    return tuple(threads.values())


def render_ledger(ledger: Sequence[Thread], *, next_episode_number: int) -> str:
    """Readable open-thread list for the next writer."""
    open_threads = _open(ledger)
    if not open_threads:
        return "目前開放中的線索：（尚無線索）"
    due_ids = {thread.id for thread in due_threads(ledger, next_episode_number)}
    lines = ["目前開放中的線索："]
    for thread in open_threads:
        label = f"{thread.id} 主線" if thread.main else thread.id
        note = f"第 {thread.planted} 集埋下，第 {thread.touched} 集最後被碰到"
        if thread.id in due_ids:
            note += f"；到期：已超過 {DUE_AFTER_EPISODES} 集沒被碰到，本集必須推進或回收"
        lines.append(f"- {label}｜{thread.text}（{note}）")
    return "\n".join(lines)


def render_recap(rows: Sequence[Mapping[str, object]]) -> str:
    """One line per committed episode: what happened and the goal it handed on."""
    return "\n".join(
        f"第 {row['episode_number']} 集《{row['topic']}》：{row['summary']}｜交給下一集：{row['next_goal']}"
        for row in rows
    )
