import json
import os

from google import genai
from google.genai import types
from pydantic import BaseModel


class LabelRelationshipsResponse(BaseModel):
    no: int
    relationship: str


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GOOGLE_GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Google GenAI API key is required. Set GOOGLE_GENAI_API_KEY in the environment.")
        self.client = genai.Client(api_key=api_key)

    def generate_content(self, prompt: str, response_schema: BaseModel, seed: int) -> dict:
        response = self.client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0,
                seed=seed,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
            ),
        )
        assert response.text is not None
        return json.loads(response.text)
