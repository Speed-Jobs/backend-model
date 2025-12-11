"""Helper Utilities for Tools"""

import json
import re


def extract_json_from_response(response_text: str) -> dict:
    """
    Extract JSON from LLM response

    LLM responses often include markdown code blocks (```json ... ```),
    text explanations, or formatting errors. This function extracts
    the pure JSON object from various response formats.

    Args:
        response_text: Raw LLM response text

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
    """
    try:
        # 1. Remove markdown code blocks (```json ... ``` or ``` ... ```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # 2. Find JSON starting with curly brace
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        # 3. Try parsing as-is
        return json.loads(response_text)

    except json.JSONDecodeError as e:
        print(f"[JSON Extraction] Failed to parse: {e}")
        print(f"[JSON Extraction] Raw response: {response_text[:500]}...")
        raise
