"""An A2UI surface action reaches the ADK agent.

The A2UI middleware (``@ag-ui/a2ui-middleware``) reports a click on a rendered
surface by appending a synthetic ``log_a2ui_event`` assistant call and its tool
result to the run's messages: the call's arguments are the whole ``userAction``
and the result is the middleware's report of it. ADK never made that call, so
the result is not a FunctionResponse it can accept; the action has to reach
the model as the user's turn, the way a ``confirm_changes`` decision does.
"""

import json

import pytest

from ag_ui.core import (
    AssistantMessage,
    EventType,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from ag_ui_adk import ADKAgent
from ag_ui_adk.session_manager import SessionManager
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.sessions import InMemorySessionService

from tests.hitl_helpers import (
    ScriptedLlm,
    collect,
    content_text,
    run_finished,
    run_input,
)

THREAD = "t-a2ui-action"
ASK = UserMessage(id="u-1", role="user", content="Compare three hotels in Paris")
BOOK_ACTION = {
    "name": "book_hotel",
    "surfaceId": "hotel-comparison",
    "sourceComponentId": "book-button-2",
    "timestamp": "2026-10-15T09:00:00.000Z",
    "context": {"hotelId": "le-meurice", "nights": 2},
}
# The report the middleware's formatUserActionResult writes for BOOK_ACTION.
BOOK_REPORT = (
    'User performed action "book_hotel" on surface "hotel-comparison" '
    '(component: book-button-2). Context: {"hotelId":"le-meurice","nights":2}'
)


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset_instance()
    yield
    SessionManager.reset_instance()


def _agent(store: InMemorySessionService, *, resumable: bool = False):
    llm = ScriptedLlm(model="scripted")
    root = LlmAgent(name="hotel_agent", model=llm)
    if resumable:
        agent = ADKAgent.from_app(
            App(
                name="hotel_app",
                root_agent=root,
                resumability_config=ResumabilityConfig(is_resumable=True),
            ),
            user_id="test_user",
            session_service=store,
        )
    else:
        agent = ADKAgent(
            adk_agent=root,
            app_name="hotel_app",
            user_id="test_user",
            session_service=store,
        )
    return agent, llm


def _surface_turn():
    """The turn that rendered the surface, as the client holds it."""
    return [
        ASK,
        AssistantMessage(id="a-1", role="assistant", content="Here are three hotels."),
    ]


def _action_pair(suffix: str = "1", report: str = BOOK_REPORT):
    """The synthetic pair the middleware appends, in its own shapes."""
    call_id = f"call-log-{suffix}"
    return [
        AssistantMessage(
            id=f"a-log-{suffix}",
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id=call_id,
                    type="function",
                    function=FunctionCall(name="log_a2ui_event", arguments=json.dumps(BOOK_ACTION)),
                )
            ],
        ),
        ToolMessage(id=f"t-log-{suffix}", role="tool", tool_call_id=call_id, content=report),
    ]


def _assert_well_formed(events):
    assert events[0].type == EventType.RUN_STARTED
    assert not [event for event in events if event.type == EventType.RUN_ERROR], [
        event.message for event in events if event.type == EventType.RUN_ERROR
    ]
    run_finished(events)


def _reply_text(events):
    return "".join(
        event.delta for event in events if event.type == EventType.TEXT_MESSAGE_CONTENT
    )


async def _render(agent):
    """Turn 1: the agent answers the question (the surface is the client's)."""
    events = await collect(agent, run_input(THREAD, "run-1", [ASK]))
    _assert_well_formed(events)


class TestA2UIActionReachesTheAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("resumable", [False, True])
    async def test_a_surface_action_is_the_users_turn(self, resumable):
        agent, llm = _agent(InMemorySessionService(), resumable=resumable)
        await _render(agent)

        events = await collect(
            agent, run_input(THREAD, "run-2", _surface_turn() + _action_pair())
        )

        _assert_well_formed(events)
        assert llm.turn_count == 2, "the action must start one model turn"
        assert content_text(llm.last_contents[-1]) == BOOK_REPORT
        assert llm.last_contents[-1].role == "user"
        assert _reply_text(events) == "Done."

    @pytest.mark.asyncio
    async def test_the_action_is_never_sent_to_adk_as_a_function_response(self):
        agent, llm = _agent(InMemorySessionService())
        await _render(agent)

        await collect(agent, run_input(THREAD, "run-2", _surface_turn() + _action_pair()))

        session = await agent._session_manager._find_session_by_thread_id(
            "hotel_app", "test_user", THREAD
        )
        responses = [
            part.function_response
            for event in session.events
            for part in (event.content.parts if event.content else [])
            if part.function_response is not None
        ]
        assert responses == []
        users = [
            content_text(event.content)
            for event in session.events
            if event.author == "user" and event.content
        ]
        assert users[-1] == BOOK_REPORT

    @pytest.mark.asyncio
    async def test_each_action_is_delivered_once(self):
        agent, llm = _agent(InMemorySessionService())
        await _render(agent)
        first = _surface_turn() + _action_pair("1")
        await collect(agent, run_input(THREAD, "run-2", first))
        assert llm.turn_count == 2

        # The client re-sends its history (without the middleware's pair, which
        # it never held) together with a second click's pair.
        replies = [AssistantMessage(id="a-2", role="assistant", content="Done.")]
        events = await collect(
            agent,
            run_input(THREAD, "run-3", _surface_turn() + replies + _action_pair("2")),
        )

        _assert_well_formed(events)
        assert llm.turn_count == 3
        assert content_text(llm.last_contents[-1]) == BOOK_REPORT

    @pytest.mark.asyncio
    async def test_an_answered_action_in_the_history_starts_no_turn(self):
        agent, llm = _agent(InMemorySessionService())
        await _render(agent)
        answered = _surface_turn() + _action_pair() + [
            AssistantMessage(id="a-2", role="assistant", content="Booked.")
        ]

        events = await collect(agent, run_input(THREAD, "run-2", answered))

        _assert_well_formed(events)
        assert llm.turn_count == 1

    @pytest.mark.asyncio
    async def test_a_report_without_text_reaches_the_agent_as_the_action_itself(self):
        agent, llm = _agent(InMemorySessionService())
        await _render(agent)

        events = await collect(
            agent,
            run_input(THREAD, "run-2", _surface_turn() + _action_pair(report="")),
        )

        _assert_well_formed(events)
        text = content_text(llm.last_contents[-1])
        assert json.loads(text[text.index("{"):]) == BOOK_ACTION
