"""A turn root starts its own trace unless something deliberately parented it.

The failure this guards against, observed on cluster-zwtpw: a chat turn was
interrupted (the operator pressed Stop, so the SSE client went away and
``on_session_end`` never fired). The next prompt in that session superseded
the stale root, which ends and exports it but leaves the ContextVar parent
entry alone on purpose (``_finalize_orphan``, #105). From then on every new
session's root was created with that dead span as its parent, inherited its
*trace id*, and landed in a trace stamped hours earlier under someone else's
session id -- so the turn had no trace of its own to find.

Two independent guards, because either alone would have prevented it:
a root does not take an ambient parent, and no stack ever hands back a span
that has already ended.
"""

from hermes_otel.hooks import (
    on_post_llm_call,
    on_pre_llm_call,
    on_session_end,
    on_session_start,
)


def _turn(session_id, message="hello"):
    """One ordinary turn, start to end."""
    on_session_start(session_id=session_id, model="gpt-4", platform="api_server")
    on_pre_llm_call(
        session_id=session_id,
        user_message=message,
        conversation_history=[],
        is_first_turn=True,
        model="gpt-4",
        platform="api_server",
    )
    on_post_llm_call(
        session_id=session_id,
        user_message=message,
        assistant_response="hi",
        conversation_history=[],
        model="gpt-4",
        platform="api_server",
    )
    on_session_end(
        session_id=session_id,
        completed=True,
        interrupted=False,
        model="gpt-4",
        platform="api_server",
    )


def _roots_of(exporter, name="agent"):
    return [s for s in exporter.get_finished_spans() if s.name == name]


class TestRootTraceIsolation:
    def test_a_new_sessions_root_is_not_parented_to_an_abandoned_one(self, inmemory_otel_setup):
        """The bug, end to end: two sessions, two traces.

        The first session's turn is interrupted -- no on_session_end -- and a
        second prompt supersedes its root. The second session must still be
        its own trace, not a child of the first.
        """
        exporter, plugin = inmemory_otel_setup

        # A turn that is interrupted: started, never ended the ordinary way.
        on_session_start(session_id="s_interrupted", model="gpt-4", platform="api_server")
        on_pre_llm_call(
            session_id="s_interrupted",
            user_message="one",
            conversation_history=[],
            is_first_turn=True,
            model="gpt-4",
            platform="api_server",
        )
        # The next prompt in the same session closes the stale root
        # (hermes.turn.final_status=superseded) and exports it.
        on_pre_llm_call(
            session_id="s_interrupted",
            user_message="two",
            conversation_history=[],
            is_first_turn=False,
            model="gpt-4",
            platform="api_server",
        )

        # A different conversation entirely.
        _turn("s_fresh")

        roots = _roots_of(exporter)
        fresh = roots[-1] if roots else None
        assert fresh is not None, "the second session never exported a root"
        # The whole point: its own trace, and no parent at all.
        assert fresh.parent is None, (
            "a fresh session's root was parented to another session's span, "
            "so its turn lands in that session's trace"
        )

        superseded = [
            s
            for s in exporter.get_finished_spans()
            if s.attributes.get("hermes.turn.final_status") == "superseded"
        ]
        assert superseded, "the interrupted root should have been superseded"
        assert fresh.context.trace_id != superseded[0].context.trace_id

    def test_an_ended_span_is_never_offered_as_a_parent(self, inmemory_otel_setup):
        """The second guard, at the level it lives at.

        A reference can outlive its span: the ContextVar pop is best-effort
        across threads, and the sweep paths leave it in place deliberately.
        """
        _exporter, plugin = inmemory_otel_setup

        span = plugin.start_span(name="agent", key="session:s_gone", kind="agent")
        plugin.spans.push_parent(span, session_id="s_gone")
        assert plugin.spans.get_current_parent("s_gone") is span

        plugin.end_span("session:s_gone")

        # Still referenced by both stacks, and no longer a candidate on either.
        assert plugin.spans.get_current_parent("s_gone") is None
        assert plugin.spans.get_current_parent() is None
        assert plugin.spans.get_current_parent("s_never_seen") is None

    def test_a_live_parent_is_still_used(self, inmemory_otel_setup):
        """Ordinary nesting must be untouched: the llm span sits under its turn."""
        exporter, _plugin = inmemory_otel_setup

        _turn("s_nested")

        spans = exporter.get_finished_spans()
        roots = [s for s in spans if s.parent is None]
        children = [s for s in spans if s.parent is not None]
        assert len(roots) == 1, "a single turn must produce exactly one root"
        assert children, "the turn's sub-spans must still nest under it"
        assert all(c.context.trace_id == roots[0].context.trace_id for c in children)
