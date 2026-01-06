from typing import Any, Optional
from pydantic import BaseModel, HttpUrl, Field, computed_field

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: dict[str, HttpUrl] # role -> agent URL
    config: dict[str, Any]


class UsageStats(BaseModel):
    """Tracks usage statistics during a conversation."""
    iterations: int = 0
    tool_calls_by_type: dict[str, int] = Field(default_factory=dict)
    context_length: int = 0
    response_length: int = 0

    @computed_field
    @property
    def tool_calls(self) -> int:
        return sum(self.tool_calls_by_type.values())

    def record_tool_call(self, tool_name: str):
        self.tool_calls_by_type[tool_name] = (
                self.tool_calls_by_type.get(tool_name, 0) + 1
        )

    def record_message(self, sent: str, received: str):
        self.context_length += len(sent)
        self.response_length += len(received)


class ConversationState(BaseModel):
    """Tracks state during a conversation with the purple agent."""
    trace: list[Any] = Field(default_factory=list)
    retrieved_resources: dict[str, Any] = Field(default_factory=dict)
    usage: UsageStats = Field(default_factory=UsageStats)


class TaskResult(BaseModel):
    """Result from running a single task."""
    final_answer: Optional[str]
    retrieved_fhir_resources: dict[str, Any]
    trace: list[Any]
    usage: Optional[dict[str, Any]]
    error: Optional[str] = None


class FHIRAgentBenchResult(BaseModel):
    """Result from medical benchmark evaluation."""

    # Summary metrics
    total_tasks: int
    correct_answers: int
    accuracy: float
    avg_precision: float
    avg_recall: float
    f1_score: float
    time_used: float

    # Per-task details
    task_results: dict[str, Any]  # question_id -> {correct, precision, recall, answer, ...}
