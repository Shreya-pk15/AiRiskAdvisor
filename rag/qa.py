"""
Grounded Question Answering (QA) Module.

Combines retrieved document context with the user question and executes a strictly
grounded system prompt against the LLM API to prevent hallucination.
"""

import os
from typing import List, Dict, Any, Tuple
from utils.helpers import format_source_attribution


SYSTEM_PROMPT_TEMPLATE = """You are an AI Project Intelligence & Risk Advisor assistant.
Your job is to answer user questions about software project artifacts strictly based on the provided project context below.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the information explicitly stated in the provided project context.
2. Do NOT use outside knowledge, guess, or invent any facts, deadlines, tasks, or team members.
3. If the answer cannot be found in the provided project context, reply EXACTLY with:
   "I could not find enough information in the uploaded project documents to answer this."
4. Be concise, direct, clear, and highlight specific deliverables, risks, or tasks when mentioned in the context.

--- PROJECT CONTEXT ---
{context_text}
-----------------------

USER QUESTION: {question}

GROUNDED ANSWER:"""


class AnswerGenerator:
    """Handles grounded QA generation using Google Gemini API or graceful fallback."""

    def __init__(self, api_key: str = None, model_name: str = "gemini-3.6-flash"):
        """
        Initialize LLM client.

        Args:
            api_key (str, optional): Gemini API key. Defaults to GEMINI_API_KEY or LLM_API_KEY env vars.
            model_name (str): Model name to use (default: gemini-3.6-flash).
        """

        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")

        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def _get_client(self, api_key: str = None):
        """Lazy load Gemini client using google-genai SDK."""
        key_to_use = api_key or self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")

        if not key_to_use or key_to_use == "your_gemini_api_key_here":
            return None

        if self._client is None or key_to_use != self.api_key:
            try:
                from google import genai
                self.api_key = key_to_use
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini client: {e}")
                return None
        return self._client

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        api_key: str = None
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """
        Generate a grounded answer for a user query based on retrieved text chunks.

        Args:
            query (str): User question.
            retrieved_chunks (List[Dict[str, Any]]): Retrieved relevant text chunks.
            api_key (str, optional): Optional override API key.

        Returns:
            Tuple[str, List[str], List[Dict[str, Any]]]:
                - answer (str)
                - source_list (List[str])
                - retrieved_chunks (List[Dict[str, Any]])
        """
        # Extract unique sources
        sources = sorted(list({
            chunk.get("metadata", {}).get("source", "Unknown Document")
            for chunk in retrieved_chunks
        }))

        # Handle case where no chunks were retrieved from ChromaDB
        if not retrieved_chunks:
            fallback_ans = "I could not find enough information in the uploaded project documents to answer this."
            return fallback_ans, [], []

        # Build context block from retrieved chunks
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            src = chunk.get("metadata", {}).get("source", "Unknown")
            context_blocks.append(f"[Chunk {idx} | Source: {src}]\n{chunk['text']}")
        
        context_text = "\n\n".join(context_blocks)
        prompt = SYSTEM_PROMPT_TEMPLATE.format(context_text=context_text, question=query)

        client = self._get_client(api_key=api_key)

        if client is None:
            # -----------------------------------------------------------------
            # 100% FREE LOCAL OFFLINE EXTRACTOR MODE (No API Key Required)
            # Extracts relevant sentences directly from retrieved document chunks.
            # -----------------------------------------------------------------
            extracted_points = []
            query_keywords = [w.lower() for w in query.split() if len(w) > 3]

            for chunk in retrieved_chunks:
                text_lines = [line.strip() for line in chunk["text"].split("\n") if line.strip()]
                for line in text_lines:
                    line_lower = line.lower()
                    # Check if line matches any query keyword or key project signals
                    if any(kw in line_lower for kw in query_keywords) or any(sig in line_lower for sig in ["objective", "deliverable", "blocked", "status", "risk", "deadline", "task"]):
                        if line not in extracted_points:
                            extracted_points.append(line)

            if extracted_points:
                synthesis = "📌 **Extracted Grounded Project Insights (Local Mode):**\n\n"
                synthesis += "\n".join([f"• {pt}" for pt in extracted_points[:6]])
            else:
                # Present top chunk snippets directly
                snippets = [f"• {chunk['text'][:250]}..." for chunk in retrieved_chunks[:3]]
                synthesis = "📌 **Relevant Document Context (Local Mode):**\n\n" + "\n\n".join(snippets)

            synthesis += "\n\n*(Note: Running in 100% free local mode. Add a Gemini API key to enable full LLM phrasing.)*"
            return synthesis, sources, retrieved_chunks

        # Try models in order of active availability
        models_to_try = [
            self.model_name,
            "gemini-3.6-flash",
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemini-1.5-flash-latest"
        ]
        last_error = None

        for model in list(dict.fromkeys(models_to_try)):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip(), sources, retrieved_chunks
            except Exception as e:
                last_error = e
                # If permission denied (403), fallback to Local Extractor
                if "PERMISSION_DENIED" in str(e) or "403" in str(e):
                    break
                continue

        # Fallback to Local Extractor mode if API fails or returns 403
        extracted_points = []
        query_keywords = [w.lower() for w in query.split() if len(w) > 3]

        for chunk in retrieved_chunks:
            text_lines = [line.strip() for line in chunk["text"].split("\n") if line.strip()]
            for line in text_lines:
                line_lower = line.lower()
                if any(kw in line_lower for kw in query_keywords) or any(sig in line_lower for sig in ["objective", "deliverable", "blocked", "status", "risk", "deadline", "task"]):
                    if line not in extracted_points:
                        extracted_points.append(line)

        synthesis = "📌 **Extracted Grounded Project Insights (Local Mode):**\n\n"
        if extracted_points:
            synthesis += "\n".join([f"• {pt}" for pt in extracted_points[:6]])
        else:
            snippets = [f"• {chunk['text'][:250]}..." for chunk in retrieved_chunks[:3]]
            synthesis += "\n\n".join(snippets)

        synthesis += f"\n\n*(Note: Cloud API call returned {type(last_error).__name__}. Falling back to 100% free local extraction.)*"
        return synthesis, sources, retrieved_chunks



