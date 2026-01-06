# A2A Agent Template

A MVP for the FHIR-Agent-Bench evaluation, to test the AgentBeats submission system.

## Project Structure

```
src/
├─ fhiragentbench   # Code from FHIRAgentBench modified as needed
├─ models.py        # Pydantic models
├─ config.py        # Loads configuration variables from environment
├─ server.py        # Server setup and agent card configuration
├─ executor.py      # A2A request handling
├─ agent.py         # Your agent implementation goes here
└─ messenger.py     # A2A messaging utilities
tests/
└─ test_agent.py    # Agent tests
Dockerfile          # Docker configuration
pyproject.toml      # Python dependencies
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Running and Testing Locally

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

There is a launcher script included as launcher/client_cli.py.

To run the scenario, make sure both green and purple agents are running. Then run
```bash
python -m launcher.client_cli scenario.toml output.json
```
This will use the scenario file scenario.toml and will save the results in output.json

## Running with Docker

```bash
# Build the image
docker build --platform linux/arm64 -t fhir-green-agent .

# Run the container
docker run --env-file .env -p 9001:9001 fhir-green-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9001
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
