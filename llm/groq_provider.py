"""
Groq API Provider Implementation.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, ValidationError

from llm.base_provider import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """
    LLM Provider for Groq API using the official groq SDK.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        model = (
            model_name
            or os.getenv("GROQ_RISK_MODEL")
            or os.getenv("GROQ_MODEL_NAME")
            or "openai/gpt-oss-120b"
        )
        key = api_key or os.getenv("GROQ_API_KEY")

        env_timeout = os.getenv("AGENT_TIMEOUT_SECONDS", "30")
        try:
            default_timeout = int(env_timeout)
        except ValueError:
            default_timeout = 30

        eff_timeout = timeout if timeout is not None else default_timeout

        super().__init__(model_name=model, api_key=key, timeout=eff_timeout)
        self._client = None

    def _get_client(self):
        """Lazy initialization of Groq Client."""
        if not self.api_key or self.api_key.strip() == "" or self.api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY is not configured or invalid.")

        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("groq library is not installed.")
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize Groq Client: {exc}") from exc
        return self._client

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> str:
        """
        Generate raw text using Groq API with timeout safety.
        """
        client = self._get_client()
        eff_timeout = timeout if timeout is not None else self.timeout

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        def _call():
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content if response and response.choices else ""

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            try:
                return future.result(timeout=eff_timeout)
            except FuturesTimeoutError:
                raise TimeoutError(f"Groq API request timed out after {eff_timeout} seconds.")
            except Exception as e:
                raise RuntimeError(f"Groq API generation error: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output adhering to a Pydantic schema using Groq JSON mode.
        Automatically falls back to alternative models on 429 rate-limit errors.
        """
        # Build list of models to try: primary + fallbacks from env
        fallback_env = os.getenv("GROQ_FALLBACK_MODELS", "llama3-8b-8192,gemma2-9b-it")
        fallback_models = [m.strip() for m in fallback_env.split(",") if m.strip()]
        models_to_try = [self.model_name] + [m for m in fallback_models if m != self.model_name]

        last_error = None
        for model_id in models_to_try:
            try:
                result = self._generate_structured_with_model(
                    model_id=model_id,
                    prompt=prompt,
                    response_schema=response_schema,
                    system_instruction=system_instruction,
                    timeout=timeout,
                )
                # Success — update primary model to the working one for subsequent calls
                if model_id != self.model_name:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Rate limit on '%s'; succeeded with fallback model '%s'.",
                        self.model_name, model_id
                    )
                    self.model_name = model_id
                return result
            except RuntimeError as e:
                last_error = e
                err_str = str(e)
                # Only continue the chain on rate-limit (429) errors
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    continue
                # For any other error, raise immediately
                raise

        raise RuntimeError(
            f"Groq API structured call failed on all models {models_to_try}: {last_error}"
        ) from last_error

    def _generate_structured_with_model(
        self,
        model_id: str,
        prompt: str,
        response_schema: Type[BaseModel],
        system_instruction: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Internal: attempt structured generation with a specific model_id."""
        client = self._get_client()
        eff_timeout = timeout if timeout is not None else self.timeout

        # Build schema instructions to enforce Pydantic structure in prompt
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        system_prompt = system_instruction or "You are a helpful AI assistant."
        full_system = (
            f"{system_prompt}\n\n"
            f"CRITICAL REQUIREMENT: You MUST respond strictly with a valid JSON object matching this JSON Schema:\n"
            f"{schema_json}\n\n"
            f"Do not include any explanation or markdown formatting outside of the JSON object."
        )

        # Groq json_object mode requires the word 'json' in user message
        user_prompt = prompt
        if "json" not in user_prompt.lower():
            user_prompt = f"{user_prompt}\n\nPlease output valid JSON."

        messages = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt}
        ]

        def _call():
            # Attempt 1: native Groq json_object mode
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                raw_content = response.choices[0].message.content if response and response.choices else ""
                if raw_content and raw_content.strip():
                    return raw_content
            except Exception as e:
                err_str = str(e)
                # Re-raise rate limit errors immediately (don't silently swallow them)
                if "rate_limit_exceeded" in err_str or "429" in err_str:
                    raise
                # For json_validate_failed or other grammar issues, fall through to attempt 2
                pass

            # Attempt 2: prompt-enforced raw JSON generation (no response_format constraint)
            fallback_messages = [
                {
                    "role": "system",
                    "content": (
                        f"{system_prompt}\n\n"
                        f"CRITICAL REQUIREMENT: Output a valid JSON object matching this JSON Schema:\n"
                        f"{schema_json}\n\n"
                        f"Respond ONLY with valid JSON. Do not include markdown codeblocks or explanation."
                    ),
                },
                {"role": "user", "content": f"{prompt}\n\nOutput VALID RAW JSON ONLY."}
            ]
            response = client.chat.completions.create(
                model=model_id,
                messages=fallback_messages,
                temperature=0.1
            )
            raw_content = response.choices[0].message.content if response and response.choices else "{}"
            return raw_content

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call)
            try:
                raw_json = future.result(timeout=eff_timeout)
            except FuturesTimeoutError:
                raise TimeoutError(f"Groq API call timed out after {eff_timeout} seconds.")
            except Exception as e:
                raise RuntimeError(f"Groq API structured call failed: {e}") from e

        # Parse & validate against Pydantic schema
        parsed_dict = self._parse_and_validate_json(raw_json, response_schema)
        return parsed_dict


    def _parse_and_validate_json(self, raw_text: str, response_schema: Type[BaseModel]) -> Dict[str, Any]:
        """Clean markdown formatting and validate against Pydantic schema."""
        cleaned = raw_text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Could not parse valid JSON from Groq output: {cleaned[:100]}...") from exc
            else:
                raise ValueError(f"No JSON object found in Groq output: {cleaned[:100]}...")

        # Unwrap list-wrapped responses: some models return [{...}] instead of {...}
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]

        # Validate model
        try:
            validated = response_schema.model_validate(data)
            return validated.model_dump()
        except ValidationError as exc:
            raise ValueError(f"Groq output validation error for {response_schema.__name__}: {exc}") from exc
