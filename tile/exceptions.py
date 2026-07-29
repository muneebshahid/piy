"""Typed failures raised across agent execution boundaries."""

from tile.types.conversation import AssistantTurn


class TurnFailedError(RuntimeError):
    """Raised when a provider turn ends with an unsuccessful assistant result."""

    def __init__(self, turn: AssistantTurn) -> None:
        """Preserve the failed turn while exposing a concise public message."""

        self.turn = turn
        super().__init__(turn.error_message or "The assistant turn failed.")


class ProviderStreamProtocolError(RuntimeError):
    """Raised when a provider stream violates its terminal contract."""
