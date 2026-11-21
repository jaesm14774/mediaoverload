import pytest

from agentic.runtime.serial_episode_plan import (
    build_serial_episode_outline_prompts,
)
from agentic.runtime.serial_ledger import SERIES_MAX_EPISODES


class TextModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = []
        self.options = {}

    def chat_completion(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return self.response


def test_user_given_an_approved_story_and_character_when_planning_an_episode_then_the_outline_covers_arc_scenes_and_shot_states():
    """User Given the approved series story, character and episode direction When planning an episode Then the prompt requests causal episode beats followed by readable scenes and shots with opening, action, prop and ending states."""
    _, prompt = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a brave, absent-minded blue-gray seal with a red satchel.",
        episode_number=2,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide discovers the lantern is being blocked by salt crystals.",
        complexity="cinematic",
    )

    assert "returns a lost cloud" in prompt
    assert "red satchel" in prompt
    assert "第 2 集" in prompt
    assert "開場狀態" in prompt
    assert "道具變化" in prompt
    assert "結尾狀態" in prompt



def test_user_given_a_cinematic_episode_when_planning_shots_then_the_camera_is_locked_and_every_shot_moves_big():
    """User Given a cinematic episode When its shot list is requested Then the camera is fixed with no slow push-in, and every shot is a continuous big body action instead of sleeping or standing still."""
    _, prompt = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a blue-gray seal.",
        episode_number=1,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide finds the blocked lantern.",
        complexity="cinematic",
    )

    assert "鏡頭固定" in prompt
    assert "不要推近" in prompt
    assert "緩慢推近" not in prompt
    assert "連續、大幅度的肢體動作" in prompt
    assert "睡覺" in prompt


def _simple_outline_prompt(**overrides) -> str:
    arguments = dict(
        approved_story="Kirby wonders where the star candy comes from.",
        approved_character="Kirby is a round pink creature with red feet.",
        episode_number=3,
        episode_topic="The humming cave",
        episode_direction="Kirby follows a hum into a cave.",
        complexity="simple",
    )
    arguments.update(overrides)
    _, prompt = build_serial_episode_outline_prompts(**arguments)
    return prompt


@pytest.mark.parametrize(("beats", "count_word"), [(3, "三個節拍"), (4, "四個節拍")])
def test_user_given_a_short_episode_when_planning_then_there_is_one_beat_for_every_ten_second_clip_and_the_last_one_is_a_hook(beats, count_word):
    """User Given an episode that renders as three or four ten-second clips When the outline is planned Then it asks for exactly one short visible beat per clip, ends on an unanswered hook instead of a final resolution, and does not ask for a second per-beat picture list that repeats the beats."""
    prompt = _simple_outline_prompt(beat_count=beats)

    assert count_word in prompt
    assert "每個節拍對應一支 10 秒影片" in prompt
    assert "約 40 字" in prompt
    assert "鉤子" in prompt
    assert "不得解開長期懸念" in prompt
    assert "請輸出四部分" in prompt
    assert "每個節拍一個簡單畫面提示" not in prompt


def test_user_given_the_final_episode_of_a_five_minute_series_when_planning_then_the_outline_closes_the_story_instead_of_hooking_a_next_episode():
    """User Given the seventh and last episode of a series that stays under five minutes When the outline is planned Then it must finish the main thread, end on a resolution with no hook and no next goal, while earlier episodes still end on a hook."""
    final = _simple_outline_prompt(episode_number=SERIES_MAX_EPISODES, beat_count=3)
    earlier = _simple_outline_prompt(episode_number=SERIES_MAX_EPISODES - 1, beat_count=3)

    assert "最後一集" in final and "演完" in final
    assert "不留鉤子" in final
    assert "無：全劇完結" in final
    assert "最後一集" not in earlier
    assert "必須留下鉤子" in earlier
    assert "不得解開長期懸念" in earlier


def test_user_given_a_story_that_h3_must_film_when_planning_then_the_plot_never_depends_on_text_screens_or_real_places():
    """User Given that H3 renders pictures and cannot show readable notes or phone screens When an episode is planned Then the plot is told in visible actions and big recognisable props, never in text, notes, posters, phone messages or dialogue, and real places and people are left out."""
    prompt = _simple_outline_prompt(episode_number=1, beat_count=3)

    for unfilmable in ("字條", "海報", "手機", "對白"):
        assert unfilmable in prompt
    assert "不要靠文字" in prompt
    assert "真實的國家、地名、機構或人物" in prompt


def test_user_given_a_one_minute_episode_when_planning_then_even_the_opening_beat_is_a_fast_big_action_not_a_calm_pose():
    """User Given a one-minute episode where every clip decides whether a viewer stays When the outline is planned Then each beat, the opening included, is 2-3 continuous big body actions from the first second, and sleeping, breathing or drifting never counts as the action."""
    prompt = _simple_outline_prompt(episode_number=1, beat_count=4)

    assert "每個節拍（包含開場）" in prompt
    assert "2 到 3 個連續、大幅度的肢體動作" in prompt
    assert "第一秒就開始動" in prompt
    for static_action in ("睡覺", "呼吸", "發光", "靜靜漂浮"):
        assert static_action in prompt


def test_user_given_recurring_characters_when_planning_an_episode_then_they_are_the_cast_who_act_in_the_beats_not_optional_extras():
    """User Given a series with recurring characters When an episode is planned Then the prompt makes them the cast of the episode, on screen from the first beat and acting in every beat, so a viewer meets them instead of Kirby alone."""
    prompt = _simple_outline_prompt(
        episode_number=1, roster_text="- Dede：頭上有葉子的橘色小圓人\n- Pupo：背郵包的企鵝"
    )

    assert "Dede" in prompt and "Pupo" in prompt
    assert "每個常駐角色都要在第一個節拍入鏡" in prompt
    assert "每個節拍都有自己推進故事的動作" in prompt
    assert "可以回歸" not in prompt


def test_user_given_clips_that_continue_from_each_others_last_frame_when_planning_then_the_protagonist_is_still_centred_on_screen_at_the_end_of_every_beat():
    """User Given that every clip continues from the previous clip's last frame When the episode is planned in either mode Then every beat must end with the protagonist clearly centred on screen, because a protagonist who drifts out of the last frame can never come back."""
    simple = _simple_outline_prompt(episode_number=1, beat_count=4)
    _, cinematic = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a blue-gray seal.",
        episode_number=1,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide finds the blocked lantern.",
        complexity="cinematic",
    )

    for prompt in (simple, cinematic):
        assert "主角必須仍在畫面中央、清楚可見" in prompt
        assert "每個常駐角色也必須完整留在畫面內、分站主角左右、看得見臉" in prompt
        assert "不得離開畫面" in prompt


def test_user_given_aired_episodes_when_planning_the_next_one_then_the_recap_the_ledger_and_the_cast_are_in_the_prompt_and_the_episode_must_follow_causally():
    """User Given the recap of aired episodes, the open threads and the recurring cast When the next outline is planned Then all three are shown and the writer must link the episode to the last one with therefore or but and advance an open thread by its id."""
    prompt = _simple_outline_prompt(
        series_recap="第 2 集《港口》：Kirby 在碼頭追一頂被風吹走的帽子｜交給下一集：找出帽子的主人",
        ledger_text="目前開放中的線索：\n- T1 主線｜信到底是寫給誰的（第 1 集埋下，第 2 集最後被碰到）",
        roster_text="- 小明：戴草帽的青年",
    )

    assert "找出帽子的主人" in prompt
    assert "T1 主線" in prompt
    assert "小明" in prompt
    assert "因此" in prompt and "但是" in prompt
    assert "推進" in prompt and "線索編號" in prompt


def test_user_given_a_new_episode_when_the_outline_is_requested_then_the_ledger_section_has_a_strict_machine_readable_format():
    """User Given any episode When its outline is requested Then the prompt specifies the ledger section entries (新增, 推進, 回收, 下集目標, 出場) so the outline can be checked without a model."""
    prompt = _simple_outline_prompt()

    for entry in ("新增 主線｜", "推進 T", "回收 T", "下集目標｜", "出場｜"):
        assert entry in prompt
    assert "線索帳本" in prompt


def test_user_given_the_first_episode_when_the_outline_is_requested_then_it_is_never_taught_to_cite_a_thread_id():
    """User Given a brand-new series with no threads yet When the first outline is requested Then the ledger format only offers 新增 / 出場 / 下集目標 and says no thread id exists, so the writer cannot cite one that the program would reject."""
    prompt = _simple_outline_prompt(episode_number=1, ledger_text="目前開放中的線索：（尚無線索）")

    assert "新增 主線｜" in prompt
    assert "下集目標｜" in prompt
    for forbidden in ("推進 T", "回收 T", "T1"):
        assert forbidden not in prompt
    assert "沒有任何線索編號" in prompt


def test_user_given_a_later_episode_when_the_outline_is_requested_then_threads_added_in_that_same_outline_are_not_citable():
    """User Given an episode after the first When the ledger format is shown Then it states that a thread added in this outline has no id yet and cannot be advanced or paid in the same episode."""
    prompt = _simple_outline_prompt(
        episode_number=2,
        ledger_text="目前開放中的線索：\n- T1 主線｜信到底是寫給誰的（第 1 集埋下，第 1 集最後被碰到）",
    )

    assert "推進 T" in prompt and "回收 T" in prompt
    assert "本集新增的線索還沒有編號" in prompt


REAL_OUTLINE = """一、本集一句話：
卡比撿到發亮小鑰匙，舉給德迪看，鑰匙被風吹進空心木。

新增｜空心木的開口處有微弱光暈  
出場｜Dede  

二、四個節拍：
開場：卡比在草地廣場撿到發亮小鑰匙，舉給德迪看
問題：風把鑰匙吹進空心木
升級：卡比趴在空心木口往裡看
解決並留下鉤子：鑰匙卡在木頭深處，卡比決定明天再來

三、本集結尾狀態卡，逐項寫人物位置：
Lolo：不在場內，不變，不變，不變，不變

四、線索帳本（格式見下）：
- 新增 主線｜這把小鑰匙到底能打開哪扇門？
- 出場｜Dede
- 下集目標｜卡比必須回到空心木旁

五、每個節拍一個簡單畫面提示：
開場：卡比在草地舉起鑰匙
"""


def test_user_given_an_approved_outline_when_the_video_is_planned_then_only_the_story_reaches_the_video_planner():
    """User Given an approved outline that also carries a state card and a thread ledger When the episode video is planned Then the planner is given the story and beats only, never the bookkeeping the creator reviews separately."""
    from agentic.runtime.serial_episode_plan import episode_story_text

    story = episode_story_text(REAL_OUTLINE)

    assert "卡比在草地廣場撿到發亮小鑰匙" in story
    assert "風把鑰匙吹進空心木" in story
    assert "卡比在草地舉起鑰匙" in story
    for bookkeeping in ("線索帳本", "下集目標", "新增 主線", "新增｜", "出場｜", "狀態卡", "不變，不變"):
        assert bookkeeping not in story


def test_user_given_a_rewritten_ledger_with_a_plain_title_when_the_video_is_planned_then_the_title_is_not_left_behind():
    """User Given an outline whose rewritten ledger carries a plain title without a section number When the video is planned Then neither the ledger entries nor its title reach the planner."""
    from agentic.runtime.serial_episode_plan import episode_story_text

    outline = (
        "一、本集一句話\n卡比撿到鑰匙。\n\n二、四個節拍\n開場：卡比撿起鑰匙\n\n"
        "線索帳本\n- 新增 主線｜鑰匙能開哪扇門\n- 下集目標｜找鎖\n"
    )

    story = episode_story_text(outline)

    assert "開場：卡比撿起鑰匙" in story
    assert "線索帳本" not in story
    assert "下集目標" not in story


def test_user_given_a_story_beat_that_starts_like_a_ledger_word_when_the_video_is_planned_then_the_beat_is_kept():
    """User Given a beat or picture line that merely starts with 出場 or 新增 When the video is planned Then that story line still reaches the planner, because only real ledger entries are bookkeeping."""
    from agentic.runtime.serial_episode_plan import episode_story_text

    outline = (
        "一、本集一句話\n卡比撿到鑰匙。\n\n二、四個節拍\n"
        "出場：Dede 從樹後跳出來\n新增：一把鑰匙掉在草地上\n"
        "\n四、線索帳本\n- 出場｜Dede\n- 下集目標｜找鎖\n"
    )

    story = episode_story_text(outline)

    assert "出場：Dede 從樹後跳出來" in story
    assert "新增：一把鑰匙掉在草地上" in story
    assert "下集目標" not in story


def test_user_given_an_end_state_card_without_a_section_number_when_the_video_is_planned_then_the_card_does_not_reach_the_planner():
    """User Given an outline whose end-state card has a bold title and no section number When the video is planned Then the card is still kept out of the planner text."""
    from agentic.runtime.serial_episode_plan import episode_story_text

    outline = (
        "一、本集一句話\n卡比撿到鑰匙。\n\n二、四個節拍\n開場：卡比撿起鑰匙\n\n"
        "**結尾狀態卡**\n卡比：站在草地，不變，不變\n\n五、畫面提示\n開場：草地上的卡比\n"
    )

    story = episode_story_text(outline)

    assert "開場：卡比撿起鑰匙" in story
    assert "開場：草地上的卡比" in story
    assert "站在草地，不變" not in story


def test_user_given_text_without_outline_sections_when_it_is_planned_then_it_is_left_unchanged():
    """User Given a plain one-line premise with no outline sections When the story text is taken Then it is returned as written."""
    from agentic.runtime.serial_episode_plan import episode_story_text

    assert episode_story_text("卡比追著氣球跑") == "卡比追著氣球跑"


def test_user_given_a_one_minute_episode_when_planning_then_the_story_is_readable_from_what_is_seen_alone():
    """User Given an episode rendered as clips that continue from each other's last frame When the outline is planned Then the one-liner states only the visible want, trouble and result, every place, prop and companion is on screen from the first beat, and a prop stays where it was left."""
    prompt = _simple_outline_prompt(episode_number=1, ledger_text="")

    assert "想要什麼、遇到什麼麻煩、怎麼收尾" in prompt
    assert "長期問題只寫在線索帳本" in prompt
    assert "第一個節拍就要讓" in prompt
    assert "前後一致" in prompt
    assert "每個節拍都留在畫面裡" in prompt


def test_user_given_a_one_minute_episode_when_planning_then_one_small_reversal_is_allowed_but_the_goal_is_not_completed():
    """User Given a simple episode When it is planned Then one small reversal is allowed so the story is not flat, while the series goal still stays open."""
    prompt = _simple_outline_prompt()

    assert "一個小反轉" in prompt
    assert "不得解開長期懸念" in prompt


def test_user_given_a_series_route_when_planning_one_episode_then_only_that_episodes_part_is_performed():
    """User Given the whole route in the approved story When episode 3 is planned Then the prompt restricts the writer to episode 3's part and treats later episodes as future, so episode 1 can never act out the story's later events."""
    prompt = _simple_outline_prompt(episode_number=3)

    assert "只演第 3 集" in prompt
    assert "之後的集數是未來" in prompt


def test_user_given_a_cinematic_episode_when_planning_then_the_series_rules_still_apply():
    """User Given a cinematic episode When it is planned Then the hook, recap and own-episode-only rules apply there too, not only to simple episodes."""
    prompt = _simple_outline_prompt(
        complexity="cinematic",
        series_recap="第 1 集《信》：Kirby 送信｜交給下一集：找收信人",
    )

    assert "鉤子" in prompt
    assert "找收信人" in prompt
    assert "只演第 3 集" in prompt


def test_user_given_an_approved_outline_when_it_is_summarised_for_later_episodes_then_only_its_one_line_story_is_kept():
    """User Given an approved outline with beats and a state card When later episodes need to know what it did Then the summary is the one-line story, not the beats."""
    from agentic.runtime.serial_episode_plan import episode_summary

    outline = (
        "一、本集一句話與系列位置\nKirby 把一封信送到朋友家門口，系列從此開始。\n\n"
        "二、四個節拍\n- 開始：背起背包\n- 問題：信掉進河裡\n"
    )

    summary = episode_summary(outline)

    assert "把一封信送到朋友家門口" in summary
    assert "信掉進河裡" not in summary


def test_user_given_a_cinematic_episode_that_renders_as_four_clips_when_planning_then_the_outline_has_exactly_one_shot_per_clip_so_the_ending_is_never_cut_off():
    """User Given a cinematic episode that renders as four clips When its outline is planned Then it asks for exactly four shots, one per clip, so the result and hook the story ends on are filmed instead of silently dropped."""
    _, prompt = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a blue-gray seal.",
        episode_number=1,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide finds the blocked lantern.",
        complexity="cinematic",
        beat_count=4,
    )

    assert "恰好 4 個鏡頭" in prompt
    assert "每個鏡頭對應一支影片" in prompt


def test_user_given_any_episode_when_planning_then_every_beat_travels_across_the_screen_instead_of_tidying_in_place():
    """User Given that a clip of small in-place chores renders as a nearly still video When an episode is planned in either mode Then every beat must include a big travelling move such as running, jumping, flying, tumbling, being thrown or chasing, and in-place tidying, catching or handing over does not count."""
    simple = _simple_outline_prompt(episode_number=1, beat_count=4)
    _, cinematic = build_serial_episode_outline_prompts(
        approved_story="A seal post carrier returns a lost cloud to its ocean home.",
        approved_character="Tide is a blue-gray seal.",
        episode_number=1,
        episode_topic="The broken lighthouse signal",
        episode_direction="Tide finds the blocked lantern.",
        complexity="cinematic",
        beat_count=4,
    )

    for prompt in (simple, cinematic):
        assert "大位移" in prompt
        assert "原地整理" in prompt
        assert "不可站定、停步或擺姿勢" in prompt

