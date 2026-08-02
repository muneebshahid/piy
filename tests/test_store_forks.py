"""Fork behavior tests for the unified SQLite Store."""

from tile import Completed
from tile.store import SQLiteStore
from tile.types import AssistantTurn, UserMessage
from tests.support.store import create_session, persist_outcome, start_run


def test_fork_session_copies_all_history_with_new_envelopes() -> None:
    """Copy full history while preserving payload and originating run ids."""

    store = SQLiteStore(in_memory=True)
    try:
        create_session(store, session_id="source")
        start_run(store, session_id="source")
        persist_outcome(
            store,
            session_id="source",
            outcome=Completed(value="done"),
            history_delta=[
                UserMessage(content="hello"),
                AssistantTurn(response_id="response-1"),
            ],
        )

        fork = store.fork_session(
            source_session_id="source",
            target_session_id="fork",
        )

        source = store.get_history("source")
        copied = store.get_history(fork.id)
        assert len(copied) == len(source) == 2
        assert [item.id for item in copied] != [item.id for item in source]
        assert {item.session_id for item in copied} == {"fork"}
        assert [item.run_id for item in copied] == [item.run_id for item in source]
        assert [item.position for item in copied] == [item.position for item in source]
        assert [item.item for item in copied] == [item.item for item in source]
        assert [item.created_at for item in copied] == [
            item.created_at for item in source
        ]
        assert store.list_runs("fork") == ()
    finally:
        store.close()
