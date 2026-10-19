"""Integration tests for the orphan-turn sweep (TTL-based cleanup).

Tests inject backdated start times via the public
``plugin.register_turn(session_id, started_at=...)`` seam rather than
writing to the tracker's turn registry directly — so the tests stay
robust to internal-state refactors.
"""

import time

from _helpers import span_by_name as _span_by_name
from opentelemetry.trace import StatusCode

from hermes_otel.hooks import (
    on_post_llm_call,
    on_pre_api_request,
    on_pre_llm_call,
    on_session_end,
    on_session_start,
)
from hermes_otel.plugin_config import HermesOtelConfig


def _root_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.parent is None]


class TestOrphanSweep:
    def test_session_finalized_with_timed_out_status(self, inmemory_otel_setup):
        """A session older than root_span_ttl_ms gets finalized as timed_out."""
        exporter, plugin = inmemory_otel_setup
        # The smallest TTL the config accepts (#487: 0 is out of range), so
        # any age past a millisecond expires.
        plugin.config = HermesOtelConfig(root_span_ttl_ms=1)

        on_session_start(session_id="s_dead", model="gpt-4", platform="cli")
        # Don't call on_session_end — simulate a dropped end hook.
        time.sleep(0.005)

        # Now fire an unrelated pre_* hook on a different session — this is
        # what the PRD expects to trigger the sweep.
        on_session_start(session_id="s_live", model="gpt-4", platform="cli")

        spans = exporter.get_finished_spans()
        # The dead session must be finalized (exported).
        timed_out_spans = [
            s for s in spans if s.attributes.get("hermes.turn.final_status") == "timed_out"
        ]
        assert len(timed_out_spans) == 1
        # Must be status OK, not ERROR, so error dashboards aren't polluted.
        from opentelemetry.trace import StatusCode

        assert timed_out_spans[0].status.status_code == StatusCode.OK

    def test_sweep_finalizes_sub_spans_for_same_session(self, inmemory_otel_setup):
        """A session with active llm/api/tool sub-spans all get finalized."""
        exporter, plugin = inmemory_otel_setup
        # Use a realistic TTL + back-date the registry entry so we don't
        # accidentally finalize the session we just started.
        plugin.config = HermesOtelConfig(root_span_ttl_ms=1_000)

        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        on_pre_api_request(
            task_id="t1",
            session_id="s1",
            platform="cli",
            model="gpt-4",
            provider="openai",
            base_url="",
            api_mode="chat",
            api_call_count=1,
            message_count=5,
            tool_count=0,
            approx_input_tokens=500,
            request_char_count=2000,
            max_tokens=0,
        )

        # Back-date the turn — simulate 10s passed without a post hook.
        plugin.register_turn("s1", started_at=time.perf_counter() - 10.0)

        # Trigger the sweep directly (bypasses needing an unrelated hook).
        plugin.sweep_expired_turns()

        spans = exporter.get_finished_spans()
        names = [s.name for s in spans]
        # Both the agent (session) span and the orphaned api span must have
        # been finalized by the sweep.
        assert "agent" in names
        assert "api.gpt-4" in names
        # And the session span carries the timed_out status.
        agent = _span_by_name(spans, "agent")
        assert agent.attributes.get("hermes.turn.final_status") == "timed_out"

    def test_sweep_does_not_expire_young_sessions(self, inmemory_otel_setup):
        """With a large TTL, the sweep is a no-op."""
        exporter, plugin = inmemory_otel_setup
        plugin.config = HermesOtelConfig(root_span_ttl_ms=3_600_000)

        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        on_session_start(session_id="s2", model="gpt-4", platform="cli")

        # Two session-starts of different sessions should NOT finalize each other.
        spans = exporter.get_finished_spans()
        assert len(spans) == 0

    def test_normal_end_path_unregisters(self, inmemory_otel_setup):
        """on_session_end removes the TTL entry so it can't be swept later."""
        from hermes_otel.hooks import on_session_end

        _, plugin = inmemory_otel_setup
        plugin.config = HermesOtelConfig(root_span_ttl_ms=1)

        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        on_session_end(
            session_id="s1", completed=True, interrupted=False, model="gpt-4", platform="cli"
        )

        # A subsequent sweep is a no-op — the entry was unregistered.
        assert plugin.sweep_expired_turns() == []


class TestSweepSessionIsolation:
    def test_one_expired_one_alive(self, inmemory_otel_setup):
        """Only the expired session is finalized; the live one keeps running."""
        exporter, plugin = inmemory_otel_setup

        # Start a session with a huge TTL, then shrink the TTL artificially
        # to finalize only the old one.
        plugin.config = HermesOtelConfig(root_span_ttl_ms=3_600_000)
        on_session_start(session_id="old", model="gpt-4", platform="cli")

        # Now swap in a tiny TTL and back-date the "old" turn via the
        # public register_turn seam so only it is past the threshold.
        plugin.config = HermesOtelConfig(root_span_ttl_ms=100)
        plugin.register_turn("old", started_at=time.perf_counter() - 10.0)

        # Start a new session — its pre_* hook will trigger the sweep.
        on_session_start(session_id="new", model="gpt-4", platform="cli")

        # Only "old" gets finalized; "new" stays registered and alive.
        # We verify via the exported spans, not by inspecting internal state.
        spans = exporter.get_finished_spans()
        timed_out = [
            s for s in spans if s.attributes.get("hermes.turn.final_status") == "timed_out"
        ]
        assert len(timed_out) == 1
        # "new" session never finalizes → sweep_expired_turns returns [].
        assert plugin.sweep_expired_turns() == []


class TestStaleRootSupersededByNextTurn:
    """A turn that ends no other way must not swallow the turns after it (#467).

    ``close_failed_turn`` covers the diagnosed cause: every API attempt of the
    turn failed. These cover the rest — a turn whose ``on_session_end`` simply
    never arrives. Its root used to stay open, and the next turn's root was
    then tracked under the same ``session:<id>`` key, dropping the open span
    object somewhere nothing could ever end it: never exported, with every
    later turn of the session nested inside it, so the trace stayed in progress
    for good.
    """

    @staticmethod
    def _prompt(message, *, first=False, history=(), session_id="s1"):
        on_pre_llm_call(
            session_id=session_id,
            user_message=message,
            conversation_history=list(history),
            is_first_turn=first,
            model="gpt-4",
            platform="cli",
        )

    @staticmethod
    def _finish(message, session_id="s1"):
        on_post_llm_call(
            session_id=session_id,
            user_message=message,
            assistant_response="done",
            conversation_history=[],
            model="gpt-4",
            platform="cli",
        )
        on_session_end(
            session_id=session_id,
            completed=True,
            interrupted=False,
            model="gpt-4",
            platform="cli",
        )

    def test_next_prompt_closes_it_and_gets_its_own_trace(self, inmemory_otel_setup):
        exporter, _ = inmemory_otel_setup
        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        self._prompt("hi", first=True)
        # No on_session_end for that turn: not an API failure, just a hook
        # that never came. The session stays live and the next prompt arrives.
        self._prompt("again", history=[{"role": "user", "content": "hi"}])
        self._finish("again")

        stale, ok = sorted(_root_spans(exporter), key=lambda s: s.start_time)
        # The whole point: two turns, two traces.
        assert stale.context.trace_id != ok.context.trace_id
        assert stale.attributes["hermes.turn.final_status"] == "superseded"
        # A swept root is not the operation's error (per PRD).
        assert stale.status.status_code == StatusCode.OK
        assert ok.attributes["hermes.turn.final_status"] == "completed"
        assert ok.attributes["hermes.turn.number"] == 2

    def test_the_stale_turns_children_are_swept_into_its_own_trace(self, inmemory_otel_setup):
        exporter, _ = inmemory_otel_setup
        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        self._prompt("hi", first=True)
        self._prompt("again", history=[{"role": "user", "content": "hi"}])
        self._finish("again")

        stale, ok = sorted(_root_spans(exporter), key=lambda s: s.start_time)
        llm = [s for s in exporter.get_finished_spans() if s.name.startswith("llm.")]
        # One llm span per turn, each inside its own trace.
        assert {s.context.trace_id for s in llm} == {
            stale.context.trace_id,
            ok.context.trace_id,
        }
        swept = [s for s in llm if s.context.trace_id == stale.context.trace_id]
        assert [s.attributes["hermes.span.sweep_reason"] for s in swept] == ["superseded"]

    def test_the_turn_that_owns_the_root_is_untouched(self, inmemory_otel_setup):
        exporter, plugin = inmemory_otel_setup
        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        # The root on_session_start just opened belongs to the turn about to
        # run, not to a previous one, so the turn's own prompt must not close
        # it. Nor may an unknown session invent one.
        assert plugin.close_stale_turn("s1") is False
        assert plugin.close_stale_turn("unknown") is False
        self._prompt("hi", first=True)
        self._finish("hi")

        (root,) = _root_spans(exporter)
        assert root.attributes["hermes.turn.final_status"] == "completed"

    def test_a_continuation_turn_leaking_does_not_swallow_the_next(self, inmemory_otel_setup):
        """The gateway's shape: no ``on_session_start``, roots synthesized.

        A bridge that adopts an already-running Hermes never sees turn 1's
        ``on_session_start``, so every root here comes from the lazy fallback
        in ``pre_llm_call``. A leak on one of those must not swallow the rest
        either.
        """
        exporter, _ = inmemory_otel_setup
        self._prompt("first", history=[{"role": "user", "content": "first"}])
        self._finish("first")
        # Turn 2's root is synthesized, and its on_session_end never comes.
        self._prompt("second", history=[{"role": "user", "content": "first"}] * 2)
        self._prompt("third", history=[{"role": "user", "content": "first"}] * 3)
        self._finish("third")

        first, stale, ok = sorted(_root_spans(exporter), key=lambda s: s.start_time)
        assert len({r.context.trace_id for r in (first, stale, ok)}) == 3
        assert first.attributes["hermes.turn.final_status"] == "completed"
        assert stale.attributes["hermes.turn.final_status"] == "superseded"
        assert ok.attributes["hermes.turn.final_status"] == "completed"

    def test_nothing_to_close_when_the_root_is_already_gone(self, inmemory_otel_setup):
        _, plugin = inmemory_otel_setup
        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        self._prompt("hi", first=True)
        # The turn claimed the root, but the root has since been ended by
        # another path (a session boundary, the TTL sweep, shutdown). There is
        # nothing left to supersede.
        plugin.end_span("session:s1")
        assert plugin.close_stale_turn("s1") is False

    def test_a_healthy_second_turn_is_not_superseded(self, inmemory_otel_setup):
        exporter, _ = inmemory_otel_setup
        on_session_start(session_id="s1", model="gpt-4", platform="cli")
        self._prompt("hi", first=True)
        self._finish("hi")
        self._prompt("again", history=[{"role": "user", "content": "hi"}])
        self._finish("again")

        roots = _root_spans(exporter)
        assert len(roots) == 2
        assert [r.attributes["hermes.turn.final_status"] for r in roots] == [
            "completed",
            "completed",
        ]
        assert len({r.context.trace_id for r in roots}) == 2
