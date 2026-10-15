"""Public default contract; the real read-only database boundary is probed live."""
import inspect

from agentic.tools.context_services import NewsContextService
from scripts.run_longvideo_production_e2e import parse_args


def test_user_given_the_normal_news_entry_when_no_pool_limit_is_overridden_then_the_latest_5000_rows_are_requested():
    """User Given normal news-driven callers When they use the public selection default Then its candidate limit is the explicitly required 5000 while the existing seven-day eligibility window remains."""
    parameters = inspect.signature(NewsContextService.get_random_news).parameters
    assert parameters["limit"].default == 5000
    assert parameters["lookback_days"].default == 7


def test_user_given_the_longvideo_news_entry_when_no_limit_is_supplied_then_it_uses_the_same_5000_candidate_default():
    """User Given the production long-video CLI When its news limit is not overridden Then it requests the same latest 5000 candidate window as normal news selection."""
    assert parse_args([]).news_limit == 5000
