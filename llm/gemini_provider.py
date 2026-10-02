"""
Google Gemini API Provider Implementation.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError

from llm.base_provider import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    LLM Provider for Google Gemini models using the official google-genai SDK.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        model = (
            model_name
            or os.getenv("GEMINI_SCOPE_MODEL")
            or os.getenv("GEMINI_MODEL_NAME")
            or "gemini-3.6-flash"
        )
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        
        env_timeout = os.getenv("AGENT_TIMEOUT_SECONDS", "30")
        try:
            default_timeout = int(env_timeout)
        except ValueError:
            default_timeout = 30
            
        eff_timeout = timeout if timeout is not None else default_timeout

        super().__init__(model_name=model, api_key=key, timeout=eff_timeout)
        self._client = None

    def _get_client(self):
        """Lazy initialization of google-genai Client."""
        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY is not configured or invalid.")

        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("google-genai library is not installed.")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Gemini Client: {exc}") from exc
        return self._client

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Generate text response with timeout safety.
        """
        client = self._get_client()
        eff_timeout = timeout if timeout is not None else self.timeout

        def _call():
            config = {}
            if system_instruction:
                config["system_instruction"] = system_instruction
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config if config else None
            )
            return response.text if response and response.text else ""

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            try:
                return future.result(timeout=eff_timeout)
            except FuturesTimeoutError:
                raise TimeoutError(f"Gemini API request timed out after {eff_timeout} seconds.")
            except Exception as e:
                raise RuntimeError(f"Gemini API generation error: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output adhering to a Pydantic schema with timeout safety.
        """
        client = self._get_client()
        eff_timeout = timeout if timeout is not None else self.timeout

        def _call():
            from google.genai import types

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            )
            if system_instruction:
                config.system_instruction = system_instruction

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            raw_text = response.text if response and response.text else "{}"
            return raw_text

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                raw_json = future.result(timeout=eff_timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Gemini API call timed out after {eff_timeout} seconds.")
        except Exception as e:
            # If native response_schema fails (e.g. SDK version incompatibilities), fallback to prompt-enforced JSON
            raw_json = self._fallback_json_generate(prompt=prompt, system_instruction=system_instruction, timeout=eff_timeout)

        # Parse JSON
        parsed_dict = self._parse_and_validate_json(raw_json, response_schema)
        return parsed_dict

    def _fallback_json_generate(self, prompt: str, system_instruction: Optional[str], timeout: int) -> str:
        """Fallback method when structured schema enforcement is unsupported by model/SDK version."""
        json_prompt = f"{prompt}\n\nIMPORTANT: Respond with VALID JSON ONLY, strictly matching the required schema."
        return self.generate_text(prompt=json_prompt, system_instruction=system_instruction, timeout=timeout)

    def _parse_and_validate_json(self, raw_text: str, response_schema: Type[BaseModel]) -> Dict[str, Any]:
        """Clean markdown JSON wrappers and parse/validate against Pydantic schema."""
        cleaned = raw_text.strip()

        # Strip markdown ```json ... ``` tags
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Search for JSON object within text
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Could not parse valid JSON from LLM response: {cleaned[:100]}...") from exc
            else:
                raise ValueError(f"No JSON object found in LLM response: {cleaned[:100]}...")

        # Unwrap list-wrapped responses: some models return [{...}] instead of {...}
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]

        # Validate with Pydantic
        validated_model = response_schema.model_validate(data)
        return validated_model.model_dump()
