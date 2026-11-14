"""Without session tracking, a thread's processed IDs do not outlive its run.

With ``track_sessions=False`` nothing untracks a session, so the in-memory
processed-message IDs would stay for every thread a process served. Each run
reads them back from session state, so they are held only while a run of the
thread is in flight, and duplicates are still found on the next run.
"""

from importlib.util import find_spec
from typing import AsyncGenerator

import pytest

from ag_ui.core import EventType, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tests.hitl_helpers import collect, run_input

APP = "policy_app"
USER = "test_user"

requires_recorded_errors = pytest.mark.skipif(
    find_spec("google.adk.workflow") is None,
    reason="ADK records a failed turn as an error event only from google-adk 2.x",
)


class CountingLlm(BaseLlm):
    """Fails its first ``failures`` calls, then answers."""

    failures: int = 0
    calls: int = 0

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("model provider unavailable")
        yield LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="Done.")]),
            partial=False,
            turn_complete=True,
        )


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def _agent(store, llm, *, track_sessions):
    return ADKAgent(
        adk_agent=LlmAgent(name="policy_agent", model=llm),
        app_name=APP,
        user_id=USER,
        session_service=store,
        use_thread_id_as_session_id=True,
        track_sessions=track_sessions,
    )


def _ask(thread: str) -> UserMessage:
    return UserMessage(id=f"{thread}-ask", role="user", content="What does my policy cover?")


def _reply(events):
    return "".join(
        event.delta for event in events if event.type == EventType.TEXT_MESSAGE_CONTENT
    )


class TestProcessedIdsLifetime:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("track_sessions", [True, False])
    async def test_only_tracked_sessions_keep_processed_ids_after_the_run(
        self, track_sessions
    ):
        llm = CountingLlm(model="counting")
        agent = _agent(InMemorySessionService(), llm, track_sessions=track_sessions)
        threads = [f"t-{n}" for n in range(20)]

        for thread in threads:
            assert _reply(await collect(agent, run_input(thread, "run-1", [_ask(thread)]))) == "Done."

        kept = [
            agent._session_manager.get_processed_message_ids(APP, thread, user_id=USER)
            for thread in threads
        ]
        if track_sessions:
            assert kept == [{_ask(thread).id} for thread in threads]
        else:
            assert kept == [set()] * len(threads)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("track_sessions", [True, False])
    async def test_resending_an_answered_turn_runs_nothing(self, track_sessions):
        llm = CountingLlm(model="counting")
        agent = _agent(InMemorySessionService(), llm, track_sessions=track_sessions)
        await collect(agent, run_input("t-1", "run-1", [_ask("t-1")]))

        resent = await collect(agent, run_input("t-1", "run-2", [_ask("t-1")]))

        assert [event.type for event in resent] == [
            EventType.RUN_STARTED,
            EventType.RUN_FINISHED,
        ]
        assert llm.calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("track_sessions", [True, False])
    async def test_a_restarted_process_still_finds_the_answered_turn(self, track_sessions):
        store = InMemorySessionService()
        llm = CountingLlm(model="counting")
        await collect(
            _agent(store, llm, track_sessions=track_sessions),
            run_input("t-1", "run-1", [_ask("t-1")]),
        )
        SessionManager.reset_instance()

        resent = await collect(
            _agent(store, llm, track_sessions=track_sessions),
            run_input("t-1", "run-2", [_ask("t-1")]),
        )

        assert _reply(resent) == ""
        assert llm.calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("track_sessions", [True, False])
    async def test_the_next_message_of_the_thread_still_runs(self, track_sessions):
        llm = CountingLlm(model="counting")
        agent = _agent(InMemorySessionService(), llm, track_sessions=track_sessions)
        first = _ask("t-1")
        await collect(agent, run_input("t-1", "run-1", [first]))
        follow_up = UserMessage(id="t-1-follow-up", role="user", content="And theft?")

        answered = await collect(agent, run_input("t-1", "run-2", [first, follow_up]))

        assert _reply(answered) == "Done."
        assert llm.calls == 2

    @requires_recorded_errors
    @pytest.mark.asyncio
    async def test_a_retry_reruns_the_failed_turn_without_tracking(self):
        llm = CountingLlm(model="counting", failures=1)
        agent = _agent(InMemorySessionService(), llm, track_sessions=False)
        failed = await collect(agent, run_input("t-1", "run-1", [_ask("t-1")]))
        assert [event.type for event in failed][-1] == EventType.RUN_ERROR

        retried = await collect(agent, run_input("t-1", "run-2", [_ask("t-1")]))

        assert _reply(retried) == "Done."
        assert llm.calls == 2


class TestHoldProcessedMessageIds:
    def test_ids_stay_until_the_threads_last_run_ends(self):
        manager = SessionManager(track_sessions=False)

        with manager.hold_processed_message_ids(APP, "t-1", user_id=USER):
            with manager.hold_processed_message_ids(APP, "t-1", user_id=USER):
                manager.mark_messages_processed(APP, "t-1", ["m-1"], user_id=USER)
            assert manager.get_processed_message_ids(APP, "t-1", user_id=USER) == {"m-1"}

        assert manager.get_processed_message_ids(APP, "t-1", user_id=USER) == set()

    def test_another_threads_run_ending_keeps_the_ids(self):
        manager = SessionManager(track_sessions=False)

        with manager.hold_processed_message_ids(APP, "t-1", user_id=USER):
            manager.mark_messages_processed(APP, "t-1", ["m-1"], user_id=USER)
            with manager.hold_processed_message_ids(APP, "t-2", user_id=USER):
                pass
            assert manager.get_processed_message_ids(APP, "t-1", user_id=USER) == {"m-1"}

    def test_tracked_sessions_keep_the_ids_after_the_run(self):
        manager = SessionManager(track_sessions=True)

        with manager.hold_processed_message_ids(APP, "t-1", user_id=USER):
            manager.mark_messages_processed(APP, "t-1", ["m-1"], user_id=USER)

        assert manager.get_processed_message_ids(APP, "t-1", user_id=USER) == {"m-1"}
