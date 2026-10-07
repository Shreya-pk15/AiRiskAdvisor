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
                        "Fallback succeeded on model '%s' (primary was '%s').",
                        model_id, self.model_name
                    )
                    self.model_name = model_id
                return result
            except Exception as e:
                last_error = e
                err_str = str(e)
                err_lower = err_str.lower()
                # If authentication failed, immediately raise
                if "api_key" in err_lower or "authentication" in err_lower or "unauthorized" in err_lower:
                    raise
                # For rate limits, 429, decommissioned models, or unparseable JSON, try the next model
                continue

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
                    cleaned_check = re.sub(r"<think>[\s\S]*?</think>", "", raw_content, flags=re.IGNORECASE).strip()
                    if "{" in cleaned_check or "[" in cleaned_check:
                        return raw_content
            except Exception as e:
                err_str = str(e)
                # Re-raise rate limit errors immediately so outer fallback loop can try next model
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
                        f"Respond ONLY with valid JSON. Do not include markdown codeblocks, thinking tags, or explanation."
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
        """Clean markdown formatting, thought tags, and validate against Pydantic schema."""
        if not raw_text or not raw_text.strip():
            try:
                return response_schema().model_dump()
            except Exception:
                raise ValueError("Groq returned empty response and schema has no defaults.")

        cleaned = raw_text.strip()

        # 1. Strip thinking / reasoning tags (e.g. <think>...</think>)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()

        # 2. Extract from markdown code fences if present
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        candidate = code_block_match.group(1).strip() if code_block_match else cleaned

        # 3. Helper to find balanced outer JSON structure
        def _extract_balanced_json(text: str) -> Optional[str]:
            for start_char, end_char in [("{", "}"), ("[", "]")]:
                start_idx = text.find(start_char)
                if start_idx != -1:
                    depth = 0
                    in_str = False
                    escape = False
                    for i in range(start_idx, len(text)):
                        c = text[i]
                        if escape:
                            escape = False
                            continue
                        if c == "\\":
                            escape = True
                            continue
                        if c == '"':
                            in_str = not in_str
                            continue
                        if not in_str:
                            if c == start_char:
                                depth += 1
                            elif c == end_char:
                                depth -= 1
                                if depth == 0:
                                    return text[start_idx : i + 1]
            return None

        # 4. Attempt parsing across extracted candidates
        data = None
        candidates_to_try = []
        balanced = _extract_balanced_json(candidate) or _extract_balanced_json(cleaned)
        if balanced:
            candidates_to_try.append(balanced)
        candidates_to_try.extend([candidate, cleaned])

        for cand in candidates_to_try:
            if not cand:
                continue
            # Attempt direct parse
            try:
                data = json.loads(cand, strict=False)
                break
            except Exception:
                pass
            # Attempt trailing commas cleanup
            try:
                sanitized = re.sub(r",\s*([\]}])", r"\1", cand)
                data = json.loads(sanitized, strict=False)
                break
            except Exception:
                pass

        if data is None:
            # Fallback regex search for anything with curly braces
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                try:
                    sanitized = re.sub(r",\s*([\]}])", r"\1", match.group(0))
                    data = json.loads(sanitized, strict=False)
                except Exception as exc:
                    raise ValueError(f"Could not parse valid JSON from Groq output: {cleaned[:120]}...") from exc
            else:
                raise ValueError(f"No JSON object found in Groq output: {cleaned[:120]}...")

        # Unwrap list-wrapped responses: some models return [{...}] instead of {...}
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]

        # Validate model
        try:
            validated = response_schema.model_validate(data)
            return validated.model_dump()
        except ValidationError as exc:
            raise ValueError(f"Groq output validation error for {response_schema.__name__}: {exc}") from exc
