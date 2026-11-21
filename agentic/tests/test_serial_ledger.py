import pytest

from agentic.runtime.serial_ledger import (
    DUE_AFTER_EPISODES,
    SERIES_MAX_EPISODES,
    LedgerError,
    apply_delta,
    parse_episode_delta,
    render_ledger,
    render_recap,
    replace_ledger_section,
    validate_cast_matches_story,
    validate_route_cast,
    validate_episode_delta,
)

OPENING_OUTLINE = """一、本集一句話
Kirby 在草地撿到一張沒有地址的信。

三、本集結尾狀態卡
人物位置：Kirby 站在郵筒旁。

四、線索帳本
- 新增 主線｜這封信到底是寫給誰的
- 新增｜信封角落有一個星形印記
- 下集目標｜找出星形印記是哪裡來的

五、畫面提示
1. Kirby 撿起信。
"""


def _outline(*lines: str, goal: str = "追著印記去郵局") -> str:
    body = "\n".join(lines)
    return f"一、本集一句話\n一句話。\n\n四、線索帳本\n{body}\n- 下集目標｜{goal}\n\n五、畫面提示\n1. x\n"


def _open_ledger():
    delta = parse_episode_delta(OPENING_OUTLINE)
    return apply_delta(
        (),
        delta,
        episode_number=1,
    )


def test_user_given_an_opening_outline_when_the_ledger_section_is_read_then_new_threads_and_the_hook_goal_are_recovered():
    """User Given an episode one outline with a ledger section When the outline is read Then the new threads, which one is the standing question, and the goal handed to the next episode are all recovered."""
    delta = parse_episode_delta(OPENING_OUTLINE)

    assert [(text, main) for text, main in delta.adds] == [
        ("這封信到底是寫給誰的", True),
        ("信封角落有一個星形印記", False),
    ]
    assert delta.next_goal == "找出星形印記是哪裡來的"
    assert delta.advances == () and delta.pays == ()


def test_user_given_an_outline_without_a_ledger_section_when_it_is_read_then_it_is_refused_with_a_reason_the_writer_can_fix():
    """User Given an outline that has no ledger section When it is read Then it is refused and the reason names the missing section."""
    with pytest.raises(LedgerError, match="線索帳本"):
        parse_episode_delta("一、本集一句話\n沒有帳本。\n")


def test_user_given_a_ledger_section_with_a_line_that_is_not_an_entry_when_it_is_read_then_the_line_is_named():
    """User Given a ledger section that contains free prose among its entries When it is read Then it is refused and the offending line is quoted."""
    outline = "四、線索帳本\n- 新增 主線｜問題\n所以主角繼續前進\n- 下集目標｜走\n"

    with pytest.raises(LedgerError, match="所以主角繼續前進"):
        parse_episode_delta(outline)


def test_user_given_a_ledger_section_without_exactly_one_next_goal_when_it_is_read_then_it_is_refused():
    """User Given a ledger section with no next-episode goal, or two When it is read Then it is refused because every episode must hand exactly one goal to the next."""
    with pytest.raises(LedgerError, match="下集目標"):
        parse_episode_delta("四、線索帳本\n- 新增 主線｜問題\n")
    with pytest.raises(LedgerError, match="下集目標"):
        parse_episode_delta("四、線索帳本\n- 新增 主線｜問題\n- 下集目標｜甲\n- 下集目標｜乙\n")


def test_user_given_a_series_with_no_standing_question_when_episode_one_adds_none_then_it_is_refused():
    """User Given a series whose ledger has no standing question When an episode adds only an ordinary thread Then it is refused because the series must always carry a question the viewer wants answered."""
    delta = parse_episode_delta(_outline("- 新增｜一個小線索"))

    with pytest.raises(LedgerError, match="主線"):
        validate_episode_delta(delta, (), episode_number=1, roster_names=())


def test_user_given_open_threads_when_an_episode_touches_none_of_them_then_it_is_refused():
    """User Given a ledger with open threads When the next episode neither advances nor pays any of them Then it is refused so the episode cannot be a stand-alone errand."""
    ledger = _open_ledger()
    delta = parse_episode_delta(_outline("- 新增｜新來的小麻煩"))

    with pytest.raises(LedgerError, match="推進"):
        validate_episode_delta(delta, ledger, episode_number=2, roster_names=())


def test_user_given_an_episode_that_advances_a_thread_when_it_is_validated_then_it_passes_and_the_thread_remembers_the_episode():
    """User Given an open thread When episode two advances it Then the episode is accepted and the ledger records that the thread was last touched in episode two."""
    ledger = _open_ledger()
    delta = parse_episode_delta(_outline("- 推進 T2｜印記和郵局的戳記一樣"))

    validate_episode_delta(delta, ledger, episode_number=2, roster_names=())
    after = apply_delta(ledger, delta, episode_number=2)

    touched = {thread.id: thread.touched for thread in after}
    assert touched["T2"] == 2
    assert touched["T1"] == 1


def test_user_given_an_episode_that_cites_an_unknown_thread_when_it_is_validated_then_the_valid_ids_are_listed():
    """User Given a ledger with threads T1 and T2 When an episode advances T9 Then it is refused and the open ids are listed so the writer can correct it."""
    ledger = _open_ledger()
    delta = parse_episode_delta(_outline("- 推進 T9｜不存在"))

    with pytest.raises(LedgerError, match="T1.*T2"):
        validate_episode_delta(delta, ledger, episode_number=2, roster_names=())


def test_user_given_the_standing_question_when_an_early_episode_tries_to_pay_it_off_then_it_is_refused():
    """User Given the series' standing question is open When an episode before the final one tries to pay it off Then it is refused because only the final episode closes the series."""
    ledger = _open_ledger()
    delta = parse_episode_delta(_outline("- 回收 T1｜信是寫給自己的"))

    with pytest.raises(LedgerError, match="主線"):
        validate_episode_delta(delta, ledger, episode_number=2, roster_names=())


def _final_outline(*lines: str) -> str:
    return _outline(*lines, goal="無：全劇完結")


def test_user_given_the_final_episode_when_it_pays_off_every_open_thread_then_the_series_has_nothing_left_open():
    """User Given a series on its final episode When the episode pays off the standing question and every other open thread Then it passes and the ledger has no open thread left, so the story is finished within its seven episodes."""
    ledger = _open_ledger()
    delta = parse_episode_delta(
        _final_outline("- 回收 T1｜信是寫給自己的", "- 回收 T2｜印記是郵局的暗號")
    )

    validate_episode_delta(delta, ledger, episode_number=SERIES_MAX_EPISODES, roster_names=())
    after = apply_delta(ledger, delta, episode_number=SERIES_MAX_EPISODES)

    assert [thread for thread in after if thread.paid is None] == []


def test_user_given_the_final_episode_when_a_thread_is_left_open_or_a_new_one_is_added_then_it_is_refused_with_the_reason():
    """User Given a series on its final episode When the episode leaves a thread open, or plants a new one Then it is refused and names what to fix, because a finished series cannot keep a promise to viewers."""
    ledger = _open_ledger()
    leaves_open = parse_episode_delta(_final_outline("- 回收 T1｜信是寫給自己的"))
    plants_new = parse_episode_delta(
        _final_outline("- 回收 T1｜答案", "- 回收 T2｜答案二", "- 新增｜又一個謎")
    )

    with pytest.raises(LedgerError, match="T2"):
        validate_episode_delta(leaves_open, ledger, episode_number=SERIES_MAX_EPISODES, roster_names=())
    with pytest.raises(LedgerError, match="新增"):
        validate_episode_delta(plants_new, ledger, episode_number=SERIES_MAX_EPISODES, roster_names=())


def test_user_given_a_series_that_already_aired_its_final_episode_when_another_is_planned_then_it_is_refused():
    """User Given a five-minute series whose seventh episode has aired When an eighth is planned Then it is refused, so the series cannot grow past five minutes."""
    delta = parse_episode_delta(_outline("- 新增 主線｜新問題"))

    with pytest.raises(LedgerError, match="最多"):
        validate_episode_delta(delta, (), episode_number=SERIES_MAX_EPISODES + 1, roster_names=())


def test_user_given_an_ordinary_thread_when_an_episode_pays_it_off_then_it_closes_and_stays_closed():
    """User Given an ordinary open thread When an episode pays it off Then it leaves the open list and cannot be advanced again."""
    ledger = _open_ledger()
    pay = parse_episode_delta(_outline("- 回收 T2｜印記是郵局的暗號"))
    validate_episode_delta(pay, ledger, episode_number=2, roster_names=())
    after = apply_delta(ledger, pay, episode_number=2)

    assert [thread.id for thread in after if thread.paid is None] == ["T1"]
    again = parse_episode_delta(_outline("- 推進 T2｜再提一次"))
    with pytest.raises(LedgerError, match="T1"):
        validate_episode_delta(again, after, episode_number=3, roster_names=())


def test_user_given_a_thread_untouched_for_several_episodes_when_the_next_episode_ignores_it_then_it_is_refused():
    """User Given a thread nobody touched for several episodes When the next episode advances only a different thread Then it is refused until the neglected thread is touched."""
    ledger = _open_ledger()
    delta = parse_episode_delta(_outline("- 推進 T1｜信有了新線索"))
    for number in range(2, 2 + DUE_AFTER_EPISODES):
        ledger = apply_delta(ledger, delta, episode_number=number)
    ignore_due = parse_episode_delta(_outline("- 推進 T1｜又提主線"))

    with pytest.raises(LedgerError, match="T2"):
        validate_episode_delta(
            ignore_due, ledger, episode_number=2 + DUE_AFTER_EPISODES, roster_names=()
        )
    touch_due = parse_episode_delta(_outline("- 推進 T2｜終於回到印記"))
    validate_episode_delta(
        touch_due, ledger, episode_number=2 + DUE_AFTER_EPISODES, roster_names=()
    )


def test_user_given_a_recurring_cast_when_an_episode_lists_who_is_on_screen_then_only_known_names_are_allowed():
    """User Given a roster with one recurring character When an episode lists a name outside the roster as on screen in the first beat Then it is refused and the roster is listed."""
    ledger = _open_ledger()
    known = parse_episode_delta(
        _outline("- 推進 T1｜信的筆跡出現", "- 出場｜小明")
    )
    unknown = parse_episode_delta(
        _outline("- 推進 T1｜信的筆跡出現", "- 出場｜陌生人")
    )

    validate_episode_delta(known, ledger, episode_number=2, roster_names=("小明",))
    assert known.present == ("小明",)
    with pytest.raises(LedgerError, match="小明"):
        validate_episode_delta(unknown, ledger, episode_number=2, roster_names=("小明",))


def test_user_given_an_outline_that_writes_a_recurring_character_into_the_story_when_the_ledger_forgets_to_list_them_then_it_is_refused():
    """User Given a roster with a recurring character When the outline's story has that character act but the ledger's on-screen line leaves them out Then it is refused, because their reference image would be missing from the first clip and the clip writers would draw them from memory; a story that never mentions them, or a ledger that lists them, passes."""
    story_with = "第 1 拍：Kirby 與小明一起追著信跑過廣場。"
    story_without = "第 1 拍：Kirby 一個人追著信跑過廣場。"
    listed = parse_episode_delta(_outline("- 新增 主線｜問題", "- 出場｜小明"))
    forgotten = parse_episode_delta(_outline("- 新增 主線｜問題"))

    validate_cast_matches_story(listed, story_with, roster_names=("小明",))
    validate_cast_matches_story(forgotten, story_without, roster_names=("小明",))
    with pytest.raises(LedgerError, match="小明"):
        validate_cast_matches_story(forgotten, story_with, roster_names=("小明",))


def test_user_given_a_story_step_that_names_a_recurring_character_when_the_outline_leaves_them_out_then_it_is_refused():
    """User Given a story whose step for this episode names a registered recurring character When the outline's ledger does not list them on screen Then it is refused, so a character the story promised can no longer vanish for the whole series; a step that names nobody, or a ledger that lists them, passes."""
    step = "矮屋簷／Kirby 與阿滿婆翻過屋頂／找到腳印"
    listed = parse_episode_delta(_outline("- 新增 主線｜問題", "- 出場｜阿滿婆"))
    left_out = parse_episode_delta(_outline("- 新增 主線｜問題"))

    validate_route_cast(listed, step, roster_names=("阿滿婆", "小豆"))
    validate_route_cast(left_out, "河邊／Kirby 跳過石頭", roster_names=("阿滿婆", "小豆"))
    validate_route_cast(left_out, "", roster_names=("阿滿婆",))
    with pytest.raises(LedgerError, match="阿滿婆"):
        validate_route_cast(left_out, step, roster_names=("阿滿婆", "小豆"))


def test_user_given_a_ledger_when_it_is_shown_to_the_next_writer_then_open_threads_and_due_warnings_are_readable():
    """User Given a ledger with an ignored thread When it is rendered for the next writer Then open threads are listed with their ids and the ignored one is flagged as due."""
    ledger = _open_ledger()
    delta = parse_episode_delta(_outline("- 推進 T1｜信有新線索"))
    for number in range(2, 2 + DUE_AFTER_EPISODES):
        ledger = apply_delta(ledger, delta, episode_number=number)

    text = render_ledger(ledger, next_episode_number=2 + DUE_AFTER_EPISODES)

    assert "T1" in text and "T2" in text
    assert "到期" in text.split("T2", 1)[1].splitlines()[0]
    assert "到期" not in text.split("T1", 1)[1].splitlines()[0]


def test_user_given_committed_episodes_when_the_recap_is_rendered_then_every_episode_shows_its_summary_and_hook_goal():
    """User Given three committed episodes When the recap is rendered Then each appears in order with its summary and the goal it handed on, with none dropped."""
    rows = [
        {"episode_number": number, "topic": f"主題{number}", "summary": f"摘要{number}", "next_goal": f"目標{number}"}
        for number in (1, 2, 3)
    ]

    text = render_recap(rows)

    lines = text.splitlines()
    assert len(lines) == 3
    assert "摘要2" in lines[1] and "目標2" in lines[1]


def test_user_given_a_writer_who_uses_a_colon_and_writes_none_for_an_empty_entry_when_the_ledger_is_read_then_the_entries_still_count():
    """User Given a ledger section that separates with a full-width colon and writes 無 for an empty payoff When it is read Then the goal is recovered and the empty entry adds nothing, instead of the outline being refused over punctuation."""
    outline = "四、線索帳本\n- 新增 主線｜問題\n- 回收｜無\n- 下集目標：追著郵票走\n"

    delta = parse_episode_delta(outline)

    assert delta.next_goal == "追著郵票走"
    assert delta.pays == ()
    assert delta.adds == (("問題", True),)


def test_user_given_a_writer_who_lists_the_protagonist_and_a_divider_when_the_ledger_is_read_then_neither_blocks_the_episode():
    """User Given a ledger section where the writer names the protagonist as on screen and adds a markdown divider When it is read Then the protagonist is dropped from the cast, the divider is ignored, and the real entries are kept."""
    outline = "四、線索帳本\n---\n- 新增 主線｜問題\n- 出場｜Kirby、小明\n- 下集目標｜走\n"

    delta = parse_episode_delta(outline, protagonist="Kirby")

    assert delta.present == ("小明",)
    assert delta.adds == (("問題", True),)

BROKEN_LEDGER_OUTLINE = """一、本集一句話
Kirby 在草地撿到一張沒有地址的信。

二、節拍
1. 開場：Kirby 撿起信。

四、線索帳本
- 出演｜Dede

五、畫面提示
1. Kirby 撿起信。
"""

REPAIRED_LEDGER = """四、線索帳本
- 新增 主線｜這封信到底是寫給誰的
- 下集目標｜找出收信人"""


def test_user_given_an_outline_with_a_malformed_ledger_when_only_the_ledger_is_rewritten_then_the_story_stays_as_written():
    """User Given a story whose ledger section was rejected When just the ledger is replaced Then the replaced outline passes the ledger parser and every story line is untouched."""
    repaired = replace_ledger_section(BROKEN_LEDGER_OUTLINE, REPAIRED_LEDGER)

    assert parse_episode_delta(repaired).next_goal == "找出收信人"
    assert "出演" not in repaired
    for story_line in ("Kirby 在草地撿到一張沒有地址的信。", "1. 開場：Kirby 撿起信。", "1. Kirby 撿起信。"):
        assert story_line in repaired


def test_user_given_an_outline_without_any_ledger_when_a_ledger_is_supplied_then_it_is_added():
    """User Given a story the model wrote without a ledger section When a ledger is supplied Then the outline gains it and still keeps the whole story."""
    without_ledger = "一、本集一句話\nKirby 撿到信。\n\n二、節拍\n1. 開場：撿信。\n"

    repaired = replace_ledger_section(without_ledger, REPAIRED_LEDGER)

    assert parse_episode_delta(repaired).next_goal == "找出收信人"
    assert "1. 開場：撿信。" in repaired


def test_user_given_a_rewritten_ledger_without_its_title_when_it_is_supplied_then_it_is_still_read_as_the_ledger():
    """User Given a rewritten ledger whose title the model dropped When it is supplied Then the outline still has a recognised ledger section."""
    untitled = "- 新增 主線｜這封信到底是寫給誰的\n- 下集目標｜找出收信人"

    repaired = replace_ledger_section(BROKEN_LEDGER_OUTLINE, untitled)

    assert parse_episode_delta(repaired).next_goal == "找出收信人"
