import logging
import time

import config

import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor
import fhiragentbench.tools.cache as cache_module
from fhiragentbench.utils import check_tool_credentials


logger = logging.getLogger("fhir_agent_evaluator")
logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9001, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    # Agent card config (unchanged)
    skill = AgentSkill(
        id="fhir_agent_evaluator",
        name="Evaluates FHIR agents",
        description="Evaluates medical agents interacting with FHIR databases for clinical tasks",
        tags=["clinical tasks", "medical agents", "fhir agent"],
        examples=["""
{
  "participants": {
    "purple_agent": "http://localhost:9002"
  },
  "config": {
    "num_tasks": 5,
  }
}
"""]
    )

    max_retries = 10
    delay_seconds = 10
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Checking tool access...")
            cache_module.CACHE_ENABLED = True
            check_tool_credentials()
            logger.info("Tool access verified...")
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed.")
            if attempt == max_retries:
                logger.error("Tool check failed after maximum retries. Exiting.")
                raise
            time.sleep(delay_seconds)

    # Start A2A server
    agent_card = AgentCard(
        name="FHIR Agent Evaluator",
        description="Evaluates medical agents interacting with FHIR databases for clinical tasks",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()