from typing import AsyncGenerator, List

import pytest

from ag_ui.core import EventType, ResumeEntry, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tests.hitl_helpers import (
    RC_TOOL_NAME,
    ConfirmationTool,
    build_confirmation_agent,
    collect,
    run_input,
    tool_call,
)

APP = "policy_app"
USER = "test_user"
TIMEOUT = 60


class RecordingLlm(BaseLlm):
    requests: List[List[str]] = []

    async def generate_content_async(
        self, llm_request, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(
            [
                part.text
                for content in llm_request.contents
                if content.role == "user"
                for part in content.parts or []
                if part.text
            ]
        )
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


def _agent(store, llm, *, track_sessions=False, timeout=TIMEOUT):
    return ADKAgent(
        adk_agent=LlmAgent(name="policy_agent", model=llm),
        user_id=USER,
        session_manager=SessionManager(
            session_service=store,
            session_timeout_seconds=timeout,
            use_thread_id_as_session_id=True,
            track_sessions=track_sessions,
        ),
        app_name=APP,
    )


def _key(thread: str):
    return (thread, USER, APP)


def _ask(thread: str, n: int = 1) -> UserMessage:
    return UserMessage(id=f"{thread}-{n}", role="user", content=f"Question {n} on {thread}")


def _idle(agent: ADKAgent, seconds: float) -> None:
    for key in agent._thread_last_seen:
        agent._thread_last_seen[key] -= seconds


def _reply(events):
    return "".join(
        event.delta for event in events if event.type == EventType.TEXT_MESSAGE_CONTENT
    )


def _cached_threads(agent: ADKAgent):
    return {
        "lookups": {key[0] for key in agent._session_lookup_cache},
        "verified": {key[0] for key in agent._sessions_verified_locally},
        "checked": {key[0] for key in agent._cache_checked_keys},
    }


class TestIdleThreadLookups:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("track_sessions", [True, False])
    async def test_threads_idle_past_the_timeout_leave_the_cache(self, track_sessions):
        agent = _agent(InMemorySessionService(), RecordingLlm(model="recording"), track_sessions=track_sessions)
        for n in range(20):
            await collect(agent, run_input(f"t-{n}", "run-1", [_ask(f"t-{n}")]))
        assert len(agent._session_lookup_cache) == 20

        _idle(agent, TIMEOUT + 1)
        await collect(agent, run_input("t-new", "run-1", [_ask("t-new")]))

        assert _cached_threads(agent) == {
            "lookups": {"t-new"},
            "verified": {"t-new"},
            "checked": set(),
        }
        assert set(agent._thread_last_seen) == {_key("t-new")}

    @pytest.mark.asyncio
    async def test_threads_seen_within_the_timeout_stay(self):
        agent = _agent(InMemorySessionService(), RecordingLlm(model="recording"))
        await collect(agent, run_input("t-1", "run-1", [_ask("t-1")]))

        _idle(agent, TIMEOUT - 1)
        await collect(agent, run_input("t-2", "run-1", [_ask("t-2")]))

        assert _cached_threads(agent)["lookups"] == {"t-1", "t-2"}

    @pytest.mark.asyncio
    async def test_a_dropped_thread_continues_its_own_session(self):
        store = InMemorySessionService()
        llm = RecordingLlm(model="recording", requests=[])
        agent = _agent(store, llm)
        first = _ask("t-1")
        await collect(agent, run_input("t-1", "run-1", [first]))
        _idle(agent, TIMEOUT + 1)
        await collect(agent, run_input("t-2", "run-1", [_ask("t-2")]))
        assert "t-1" not in _cached_threads(agent)["lookups"]

        follow_up = _ask("t-1", 2)
        answered = await collect(agent, run_input("t-1", "run-2", [first, follow_up]))

        assert _reply(answered) == "Done."
        assert llm.requests[-1] == ["Question 1 on t-1", "Question 2 on t-1"]
        listed = await store.list_sessions(app_name=APP, user_id=USER)
        assert sorted(session.id for session in listed.sessions) == ["t-1", "t-2"]

    @pytest.mark.asyncio
    async def test_a_thread_with_a_run_in_flight_is_kept(self):
        agent = _agent(InMemorySessionService(), RecordingLlm(model="recording"))
        await collect(agent, run_input("t-1", "run-1", [_ask("t-1")]))

        with agent._hold_thread(_key("t-1")):
            _idle(agent, TIMEOUT + 1)
            await collect(agent, run_input("t-2", "run-1", [_ask("t-2")]))
            assert _cached_threads(agent)["lookups"] == {"t-1", "t-2"}

        _idle(agent, TIMEOUT + 1)
        await collect(agent, run_input("t-3", "run-1", [_ask("t-3")]))
        assert _cached_threads(agent)["lookups"] == {"t-3"}

    @pytest.mark.asyncio
    async def test_without_a_timeout_every_thread_stays(self):
        agent = _agent(InMemorySessionService(), RecordingLlm(model="recording"), timeout=None)
        for n in range(5):
            await collect(agent, run_input(f"t-{n}", "run-1", [_ask(f"t-{n}")]))

        assert len(agent._session_lookup_cache) == 5
        assert not agent._thread_last_seen

    @pytest.mark.asyncio
    async def test_an_idle_threads_open_interrupt_leaves_and_its_resume_still_runs(self):
        tool = ConfirmationTool()
        agent, _ = build_confirmation_agent(tool, session_timeout_seconds=TIMEOUT)
        key = ("t-1", "test_user", "confirmation_app")
        ask = UserMessage(id="u-1", role="user", content="Run the dangerous action with target='foo'")
        paused = await collect(agent, run_input("t-1", "run-1", [ask]))
        interrupt_id, _ = tool_call(paused, RC_TOOL_NAME)
        assert key in agent._open_interrupts

        _idle(agent, TIMEOUT + 1)
        agent._touch_thread(("t-2", "test_user", "confirmation_app"))
        assert key not in agent._open_interrupts
        assert key not in agent._session_lookup_cache

        resumed = await collect(
            agent,
            run_input(
                "t-1",
                "run-2",
                [ask],
                resume=[ResumeEntry(interrupt_id=interrupt_id, status="resolved", payload={"confirmed": True})],
            ),
        )

        assert not [event for event in resumed if event.type == EventType.RUN_ERROR]
        assert tool.executed == 1
