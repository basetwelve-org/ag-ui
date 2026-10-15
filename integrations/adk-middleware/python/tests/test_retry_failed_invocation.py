"""A client's retry of a failed turn re-runs that turn in place.

A client retries a failed run by re-sending the same messages. The message the
failed turn answered is already processed, so without a retry path the run is
an empty RUN_STARTED/RUN_FINISHED pair and the turn stays failed. ADK re-runs a
failed invocation given its id and no new message, recovering the user content
it recorded; the middleware hands it that invocation.
"""

from typing import AsyncGenerator

import pytest

from ag_ui.core import EventType, UserMessage
from ag_ui_adk import ADKAgent
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tests.hitl_helpers import collect, run_finished, run_input

THREAD = "t-retry"
ASK = UserMessage(id="u-1", role="user", content="What does my policy cover?")


class FlakyLlm(BaseLlm):
    """Fails its first ``failures`` calls, then answers."""

    failures: int = 1
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


def _agent(store: InMemorySessionService, *, resumable: bool, failures: int = 1):
    llm = FlakyLlm(model="flaky", failures=failures)
    root = LlmAgent(name="policy_agent", model=llm)
    if resumable:
        agent = ADKAgent.from_app(
            App(
                name="policy_app",
                root_agent=root,
                resumability_config=ResumabilityConfig(is_resumable=True),
            ),
            user_id="test_user",
            session_service=store,
        )
    else:
        agent = ADKAgent(
            adk_agent=root,
            app_name="policy_app",
            user_id="test_user",
            session_service=store,
        )
    return agent, llm


async def _user_events(agent: ADKAgent):
    session = await agent._session_manager._find_session_by_thread_id(
        "policy_app", "test_user", THREAD
    )
    return [
        event
        for event in session.events
        if event.author == "user" and event.content is not None
    ]


def _reply(events):
    return "".join(
        event.delta for event in events if event.type == EventType.TEXT_MESSAGE_CONTENT
    )


class TestRetryOfAFailedTurn:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("resumable", [False, True])
    async def test_a_retry_reruns_the_failed_turn_in_place(self, resumable):
        agent, llm = _agent(InMemorySessionService(), resumable=resumable)
        failed = await collect(agent, run_input(THREAD, "run-1", [ASK]))
        assert [event.type for event in failed][-1] == EventType.RUN_ERROR

        retried = await collect(agent, run_input(THREAD, "run-2", [ASK]))

        run_finished(retried)
        assert not [event for event in retried if event.type == EventType.RUN_ERROR]
        assert _reply(retried) == "Done."
        assert llm.calls == 2, "one failed call, then exactly one retried call"
        user_events = await _user_events(agent)
        assert len(user_events) == 1, "the retry stores the visitor's message once"

    @pytest.mark.asyncio
    async def test_a_retry_that_fails_again_can_be_retried_again(self):
        agent, llm = _agent(InMemorySessionService(), resumable=False, failures=2)
        await collect(agent, run_input(THREAD, "run-1", [ASK]))
        again = await collect(agent, run_input(THREAD, "run-2", [ASK]))
        assert [event.type for event in again][-1] == EventType.RUN_ERROR

        retried = await collect(agent, run_input(THREAD, "run-3", [ASK]))

        assert _reply(retried) == "Done."
        assert llm.calls == 3
        assert len(await _user_events(agent)) == 1

    @pytest.mark.asyncio
    async def test_resending_an_answered_turn_runs_nothing(self):
        agent, llm = _agent(InMemorySessionService(), resumable=False, failures=0)
        answered = await collect(agent, run_input(THREAD, "run-1", [ASK]))
        assert _reply(answered) == "Done."

        resent = await collect(agent, run_input(THREAD, "run-2", [ASK]))

        assert [event.type for event in resent] == [
            EventType.RUN_STARTED,
            EventType.RUN_FINISHED,
        ]
        assert llm.calls == 1
