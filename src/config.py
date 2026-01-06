"""Configuration from environment variables."""

import os

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# FHIR Server URL (local HAPI FHIR)
FHIR_SERVER_URL = os.environ.get("FHIR_SERVER_URL", "http://localhost:8080/fhir")
