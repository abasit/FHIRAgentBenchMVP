import ast
import json
import logging
import os
import time
from typing import Any, Optional
from pydantic import ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

import fhiragentbench.tools.cache as cache_module
from messenger import Messenger
from fhiragentbench.tools import get_tool_definitions, get_tool
from fhiragentbench.tools.request_tools import supported_types
from fhiragentbench.utils import read_input_data, curate_input_dataset, check_tool_credentials, parse_outputs
from fhiragentbench.utils.evaluation_metrics import calculate_answer_metrics, calculate_retrieval_metrics
from models import EvalRequest, FHIRAgentBenchResult, TaskResult, ConversationState


logger = logging.getLogger("fhir_agent_evaluator")
logger.setLevel(logging.INFO)

RESPOND_ACTION_NAME = "response"
ENABLED_TOOLS = ["fhir_request_get", "lookup_medical_code"]
DEFAULT_TASKS_FILE = os.environ.get("TASKS_FILE", "data/fhiragentbench_tasks.csv")
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_NUM_TASKS = None  # None means all tasks


def tools_to_str(tools: list[dict]) -> str:
    """Convert fhiragentbench tools to JSON schema format."""
    return json.dumps(tools)


class Agent:
    required_roles: list[str] = ["purple_agent"]
    required_config_keys: list[str] = [] # All config is optional

    # Tools to expose to the purple agent

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here

        # Store defaults (can be overridden by EvalRequest.config)
        self._default_tasks_file = DEFAULT_TASKS_FILE
        self._default_max_iterations = DEFAULT_MAX_ITERATIONS
        self._default_num_tasks = DEFAULT_NUM_TASKS

        # Task data (loaded during run_eval)
        self.tasks_df = None

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        """Run the evaluation."""
        logger.info(f"Starting FHIRAgentBench evaluation: {request}")
        start_time = time.time()

        # Get config with defaults
        tasks_file = request.config.get("tasks_file", self._default_tasks_file)
        max_iterations = request.config.get("max_iterations", self._default_max_iterations)
        num_tasks = request.config.get("num_tasks", self._default_num_tasks)
        eval_model = request.config.get("eval_model", "o4-mini")

        # Get purple agent URL
        purple_agent_url = str(request.participants["purple_agent"])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation against {purple_agent_url}")
        )

        try:
            # Initialize tools
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Checking tool access...")
            )
            logger.info("Checking tool access...")
            cache_module.CACHE_ENABLED = True
            check_tool_credentials()

            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Tool access verified...")
            )
            logger.info(f"Tool access verified...")

            # Load tasks
            self._load_tasks(tasks_file, num_tasks)
            total_tasks = len(self.tasks_df)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loaded {total_tasks} tasks")
            )
            logger.info(f"Loaded {total_tasks} tasks")

            # Run all tasks
            results_df = await self._run_all_tasks(
                purple_agent_url=purple_agent_url,
                max_iterations=max_iterations,
                updater=updater,
            )

            # Run evaluation
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Running evaluation metrics...")
            )

            eval_summary = self._evaluate_results(
                results_df,
                eval_model=eval_model,
            )

            time_used = time.time() - start_time

            # Build result
            result = FHIRAgentBenchResult(
                total_tasks=eval_summary["total_questions"],
                correct_answers=eval_summary["correct_answers"],
                accuracy=eval_summary["accuracy"],
                avg_precision=eval_summary["avg_precision"],
                avg_recall=eval_summary["avg_recall"],
                f1_score=eval_summary["f1_score"],
                time_used=time_used,
                task_results=self._build_task_results(eval_summary["eval_df"]),
            )

            # Format summary
            summary = self._format_summary(result)

            # Add artifact
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result.model_dump())),
                ],
                name="Result",
            )
        except Exception as e:
            logger.error(f"Failed to run evaluation: {e}")
        finally:
            self.messenger.reset()

    async def _run_all_tasks(
            self,
            purple_agent_url: str,
            max_iterations: int,
            updater: TaskUpdater,
    ):
        """Run all tasks sequentially."""
        tasks_df = self.tasks_df.copy()
        total_tasks = len(tasks_df)

        # Track progress
        completed = 0
        succeeded = 0
        failed = 0
        start_time = time.time()

        for idx, task in enumerate(tasks_df.itertuples(index=True)):
            task_start = time.time()
            logger.info(f"[Task {idx}] Starting")

            result = await self._run_single_task(
                purple_agent_url=purple_agent_url,
                task_id=idx,
                question=task.question_with_context,
                max_iterations=max_iterations,
            )

            elapsed = time.time() - task_start
            completed += 1

            if result.error:
                failed += 1
                logger.warning(f"[Task {idx}] Failed in {elapsed:.1f}s: {result.error}")
            else:
                succeeded += 1
                logger.info(f"[Task {idx}] Completed in {elapsed:.1f}s")

            # Progress update
            total_elapsed = time.time() - start_time
            rate = completed / total_elapsed if total_elapsed > 0 else 0
            eta = (total_tasks - completed) / rate if rate > 0 else 0

            progress_msg = (
                f"Progress: {completed}/{total_tasks} "
                f"({succeeded} ok, {failed} failed) "
                f"ETA: {eta:.0f}s"
            )
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(progress_msg)
            )

            # Process and store result
            output_dict = {
                "final_answer": result.final_answer,
                "retrieved_fhir_resources": result.retrieved_fhir_resources,
                "trace": result.trace,
                "usage": result.usage,
            }
            if result.error is not None:
                output_dict["error"] = result.error

            processed = parse_outputs(output_dict)
            for key, value in processed.items():
                tasks_df.at[tasks_df.index[idx], key] = value

        return tasks_df

    async def _run_single_task(
            self,
            purple_agent_url: str,
            task_id: int,
            question: str,
            max_iterations: int,
    ) -> TaskResult:
        """Run a single task against the purple agent."""
        state = ConversationState()
        system_prompt = self._build_task_prompt()

        # Initialize trace
        state.trace.append({"role": "system", "content": system_prompt})
        state.trace.append({"role": "user", "content": question})

        # First message combines system prompt and question
        message_content = f"{system_prompt}\n\n{question}"
        is_first_message = True

        while state.usage.iterations < max_iterations:
            state.usage.iterations += 1
            logger.info(f"[Task {task_id}] Iteration {state.usage.iterations}/{max_iterations}")

            # Send message to purple agent
            try:
                logger.debug(f"[Task {task_id}] Sending:\n{message_content}")

                response_text = await self.messenger.talk_to_agent(
                    message=message_content,
                    url=purple_agent_url,
                    new_conversation=is_first_message,
                )
                is_first_message = False
                state.usage.record_message(message_content, response_text)

                logger.debug(f"[Task {task_id}] Received:\n{response_text}")

            except Exception as e:
                logger.error(f"[Task {task_id}] Communication error: {e}")
                return TaskResult(
                    final_answer=None,
                    retrieved_fhir_resources=state.retrieved_resources,
                    trace=state.trace,
                    usage=state.usage.model_dump(),
                    error=f"Communication error - {str(e)}",
                )

            # Parse response for actions
            try:
                actions = self._parse_agent_response(response_text)
                logger.debug(f"[Task {task_id}] Parsed {len(actions)} action(s)")
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"[Task {task_id}] Parse error: {e}")
                return TaskResult(
                    final_answer=None,
                    retrieved_fhir_resources=state.retrieved_resources,
                    trace=state.trace,
                    usage=state.usage.model_dump(),
                    error=f"Failed to parse response - {str(e)}",
                )

            # Process each action
            tool_outputs = []

            for action_dict in actions:
                action_name = action_dict.get("name")

                if action_name == RESPOND_ACTION_NAME:
                    content = action_dict.get("kwargs", {}).get("content", "")
                    state.trace.append({"role": "assistant", "content": content})

                    if self._is_final_answer(content):
                        logger.info(f"[Task {task_id}] Got final answer after {state.usage.iterations} iterations")
                        return TaskResult(
                            final_answer=content,
                            retrieved_fhir_resources=state.retrieved_resources,
                            trace=state.trace,
                            usage=state.usage.model_dump(),
                        )
                    else:
                        logger.info(f"[Task {task_id}] Response without final answer, prompting again")
                        tool_outputs.append(
                            "Please provide your final answer starting with 'The final answer is:'"
                        )
                elif action_name.lower() == "error":
                    return TaskResult(
                        final_answer=None,
                        retrieved_fhir_resources=state.retrieved_resources,
                        trace=state.trace,
                        usage=state.usage.model_dump(),
                        error=action_dict.get("content"),
                    )
                else:
                    # Tool call
                    tool_name = action_name
                    tool_args = action_dict.get("kwargs", {})

                    logger.info(f"[Task {task_id}] Calling tool: {tool_name}")
                    logger.debug(f"[Task {task_id}] Tool args: {tool_args}")

                    # Add tool call to trace
                    state.trace.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                            "id": None,
                            "type": "function",
                        }]
                    })

                    # Execute tool
                    try:
                        tool_output = self._execute_tool(tool_name, tool_args)

                        # Track tool calls and resources
                        state.usage.record_tool_call(tool_name)
                        state.retrieved_resources.update(tool_output)

                        tool_output_str = str(tool_output)

                        logger.debug(f"[Task {task_id}] Tool returned:\n{tool_output_str}")

                        # Add tool result to trace
                        state.trace.append({
                            "tool_call_id": None,
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_output_str,
                        })

                        tool_outputs.append(tool_output_str)

                    except Exception as e:
                        logger.error(f"[Task {task_id}] Tool {tool_name} failed: {e}")
                        return TaskResult(
                            final_answer=None,
                            retrieved_fhir_resources=state.retrieved_resources,
                            trace=state.trace,
                            usage=state.usage.model_dump(),
                            error=f"Tool execution failed - {str(e)}",
                        )

            # Combine tool outputs for next message
            message_content = "\n\n".join(tool_outputs)

        # Max iterations reached
        logger.warning(f"[Task {task_id}] Max iterations ({max_iterations}) reached")
        return TaskResult(
            final_answer=None,
            retrieved_fhir_resources=state.retrieved_resources,
            trace=state.trace,
            usage=state.usage.model_dump(),
            error="max_iterations_reached",
        )

    def _load_tasks(self, tasks_file: str, num_tasks: Optional[int]):
        """Load tasks from file."""
        self.tasks_df = read_input_data(tasks_file)

        # Limit to subset if specified
        if num_tasks is not None:
            self.tasks_df = self.tasks_df[:num_tasks].copy()

        # Add columns for results
        for col in ["agent_answer", "agent_fhir_resources", "trace", "error", "usage"]:
            self.tasks_df[col] = None

        # Add question_with_context
        all_inputs = curate_input_dataset(self.tasks_df, True)
        self.tasks_df["question_with_context"] = all_inputs

    @staticmethod
    def _build_task_prompt() -> str:
        """Build the system prompt with tool descriptions."""
        all_tools = get_tool_definitions()
        tools = [t for t in all_tools if t["function"]["name"] in ENABLED_TOOLS]

        return f"""
        Here's a list of tools you can use (you can use one tool at a time):
{tools_to_str(tools)}

Available FHIR resource types: {', '.join(supported_types)}.
You can only call on these FHIR resources types for retrieval.

To answer questions about patient data:
1. Always use the Patient FHIR ID provided in context; do not use the numeric patient ID from the question text.
2. Use lookup_medical_code to find codes for medical items, labs, and procedures
3. Use fhir_request_get to get data from the FHIR server
4. Analyze the retrieved data to answer the question

If there are multiple answers, provide all of them.
When you provide answers, make sure to provide them in the same format as they are in the retrieved data. If multiple answers are provided, provide them all in a list.
If you cannot find the answer or relevant patient data, clearly state that you cannot find the information.
Do not guess attributes; instead, use the provided tool to retrieve the data.
Do not get stuck or repeat the same action.

Please respond in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON can be either a single action or a list of actions.

Tool Call:
{{"name": "tool_name", "kwargs": {{"arg": "value"}}}}

For final response:
{{"name": "{RESPOND_ACTION_NAME}", "kwargs": {{"content": "The final answer is: ..."}}}}

IMPORTANT: The content for your final message must start with 'The final answer is:' followed by your conclusion. This is required for proper processing.
"""

    @staticmethod
    def _parse_agent_response(response_text: str) -> list:
        """Parse purple agent response to extract action(s). Always returns a list."""
        import re

        json_str = None

        # Try to extract JSON from <json>...</json> tags
        match = re.search(r'<json>\s*(.*?)\s*</json>', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Try to extract JSON from markdown code blocks ```json ... ```
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                # Try to extract from generic code blocks ``` ... ```
                match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)

        if json_str:
            parsed = json.loads(json_str)
        else:
            # Try to parse the entire response as JSON
            parsed = json.loads(response_text)

        # Normalize to list
        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]

    @staticmethod
    def _is_final_answer(content: str) -> bool:
        """Check if content contains a final answer."""
        if not content:
            return False
        return "the final answer is:" in content.lower().strip()

    @staticmethod
    def _execute_tool(tool_name: str, tool_args: dict) -> Any:
        """Execute a tool and return the result."""
        tool_function = get_tool(tool_name)
        return tool_function(**tool_args)

    @staticmethod
    def _evaluate_results(results_df, eval_model: str) -> dict:
        """Run evaluation on results DataFrame."""
        eval_df = results_df.copy()

        # Parse true_fhir_ids if string
        if "true_fhir_ids" in eval_df.columns:
            eval_df["true_fhir_ids"] = eval_df["true_fhir_ids"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {}
            )
        else:
            eval_df["true_fhir_ids"] = [{}] * len(eval_df)

        # Retrieval metrics
        print("Calculating retrieval metrics...")
        eval_df = calculate_retrieval_metrics(eval_df)

        # Answer metrics
        print("Calculating answer metrics...")
        eval_df = calculate_answer_metrics(eval_df, eval_model)

        # Summary
        total = len(eval_df)
        correct = eval_df["answer_correctness"].sum()
        avg_precision = eval_df["precision"].mean()
        avg_recall = eval_df["recall"].mean()
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

        return {
            "total_questions": total,
            "correct_answers": int(),
            "accuracy": correct / total if total > 0 else 0,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "f1_score": f1,
            "eval_df": eval_df,
        }

    @staticmethod
    def _build_task_results(eval_df) -> dict:
        """Build per-task results dictionary."""
        task_results = {}
        for _, row in eval_df.iterrows():
            task_id = str(row.get("question_id", row.name))
            task_results[task_id] = {
                "correct": int(row.get("answer_correctness", 0)),
                "precision": float(row.get("precision", 0)),
                "recall": float(row.get("recall", 0)),
                "agent_answer": row.get("agent_answer"),
                "true_answer": row.get("true_answer"),
            }
        return task_results

    @staticmethod
    def _format_summary(result: FHIRAgentBenchResult) -> str:
        """Format result as human-readable summary."""
        return f"""Medical Benchmark Results
========================
Tasks: {result.total_tasks}
Accuracy: {result.accuracy:.1%} ({result.correct_answers}/{result.total_tasks})
Precision: {result.avg_precision:.4f}
Recall: {result.avg_recall:.4f}
F1 Score: {result.f1_score:.4f}
Time: {result.time_used:.1f}s"""
