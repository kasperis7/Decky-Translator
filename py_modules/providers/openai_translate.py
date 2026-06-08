# providers/openai_translate.py
# Translation using any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, etc.)

import asyncio
import json
import logging
from typing import List

import requests

from .base import TranslationProvider, ProviderType, NetworkError, ApiKeyError

logger = logging.getLogger(__name__)

# Default endpoint – user can override via settings
DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 30  # seconds for translation; batch may take longer


class OpenAITranslateProvider(TranslationProvider):
    """
    Translation provider using any OpenAI-compatible chat completion API.

    Supports:
      - OpenAI (api.openai.com)
      - DeepSeek (api.deepseek.com)
      - Any self-hosted vLLM / Ollama / LiteLLM endpoint
    """

    # All language codes the plugin uses – we tell the LLM to translate to these
    LANGUAGE_MAP = {
        'auto': 'auto-detect',
        'en': 'English',
        'ja': 'Japanese',
        'zh-CN': 'Simplified Chinese',
        'zh-TW': 'Traditional Chinese',
        'ko': 'Korean',
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'el': 'Greek',
        'fi': 'Finnish',
        'nl': 'Dutch',
        'pl': 'Polish',
        'tr': 'Turkish',
        'uk': 'Ukrainian',
        'hi': 'Hindi',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'ro': 'Romanian',
        'bg': 'Bulgarian',
        'hr': 'Croatian',
        'cs': 'Czech',
        'hu': 'Hungarian',
        'sv': 'Swedish',
        'da': 'Danish',
    }

    SUPPORTED_LANGUAGES = list(LANGUAGE_MAP.keys())

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "",
        model: str = "",
    ):
        self._api_key = api_key
        self._endpoint = endpoint or DEFAULT_OPENAI_ENDPOINT
        self._model = model or DEFAULT_OPENAI_MODEL
        logger.debug(
            "OpenAITranslateProvider initialized (endpoint=%s, model=%s)",
            self._endpoint, self._model,
        )

    # -- config setters (called by ProviderManager / main.py) --

    def set_api_key(self, api_key: str) -> None:
        self._api_key = api_key

    def set_endpoint(self, endpoint: str) -> None:
        self._endpoint = endpoint or DEFAULT_OPENAI_ENDPOINT

    def set_model(self, model: str) -> None:
        self._model = model or DEFAULT_OPENAI_MODEL

    # -- Provider interface --

    @property
    def name(self) -> str:
        return "OpenAI Compatible"

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI

    def is_available(self, source_lang: str = "auto", target_lang: str = "en") -> bool:
        return bool(self._api_key) and target_lang in self.SUPPORTED_LANGUAGES

    def get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES.copy()

    def _lang_name(self, code: str) -> str:
        return self.LANGUAGE_MAP.get(code, code)

    # -- Core translation logic --

    def _build_system_prompt(self, source_lang: str, target_lang: str) -> str:
        src_name = self._lang_name(source_lang)
        tgt_name = self._lang_name(target_lang)

        prompt = (
            f"You are a professional translator. Translate the following text "
            f"from {src_name} to {tgt_name}. "
        )
        if source_lang == "auto":
            prompt += "Auto-detect the source language. "
        prompt += (
            "Return ONLY the translation as a JSON array of strings, "
            "one per input line, in the same order. "
            "Preserve formatting, line breaks, and special characters. "
            "If a line is already in the target language or is untranslatable "
            "(numbers, symbols, etc.), return it unchanged."
        )
        return prompt

    def _call_openai(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        """Make a single batch request to the OpenAI-compatible endpoint."""
        system_prompt = self._build_system_prompt(source_lang, target_lang)

        # Build a numbered list of texts for clear 1:1 mapping
        user_lines = []
        for i, t in enumerate(texts):
            user_lines.append(f"[{i}] {t}")
        user_message = "\n".join(user_lines)

        url = f"{self._endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,      # low temperature for consistent translations
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},  # enforce JSON output
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 401 or resp.status_code == 403:
            raise ApiKeyError("Invalid API key")
        if resp.status_code == 429:
            from .base import RateLimitError
            raise RateLimitError("Rate limited by API provider")
        if resp.status_code != 200:
            body = resp.text[:500]
            logger.error(f"OpenAI API error {resp.status_code}: {body}")
            raise NetworkError(f"API returned {resp.status_code}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        return self._parse_response(content, texts)

    def _parse_response(self, content: str, original_texts: List[str]) -> List[str]:
        """Parse the JSON response, with robust fallback."""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or surrounding text
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse OpenAI response as JSON array")
                    return original_texts
            else:
                logger.warning("Failed to parse OpenAI response")
                return original_texts

        # Handle different response shapes
        if isinstance(parsed, list):
            translations = [str(t) for t in parsed]
        elif isinstance(parsed, dict):
            # Some models return {"translations": [...]}
            arr = parsed.get("translations") or list(parsed.values())
            translations = [str(t) for t in arr] if arr else original_texts
        else:
            return original_texts

        # Pad/truncate to match input length
        while len(translations) < len(original_texts):
            translations.append(original_texts[len(translations)])
        return translations[:len(original_texts)]

    # -- Public API --

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        results = await self.translate_batch([text], source_lang, target_lang)
        return results[0] if results else text

    async def translate_batch(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        """
        Translate multiple texts in a single batch request.
        """
        if not texts:
            return texts

        if not self._api_key:
            logger.error("OpenAI API key not configured")
            return texts

        logger.debug(
            "OpenAI batch translating %d texts: %s -> %s (model=%s)",
            len(texts), source_lang, target_lang, self._model,
        )

        try:
            return await asyncio.to_thread(
                self._call_openai, texts, source_lang, target_lang
            )
        except ApiKeyError:
            raise
        except NetworkError:
            raise
        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return texts

    # -- Network probe for reachability check --

    async def test_network(self) -> tuple:
        if not self._api_key:
            return False, "API key required"

        def _probe():
            url = f"{self._endpoint.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            try:
                resp = requests.get(url, headers=headers, timeout=5)
            except requests.ConnectionError:
                return False, "Network unreachable"
            except requests.Timeout:
                return False, "Connection timed out"
            except Exception as e:
                return False, f"Probe failed: {type(e).__name__}"
            if resp.status_code == 200:
                return True, ""
            if resp.status_code in (401, 403):
                return False, "Invalid API key"
            if resp.status_code == 429:
                return False, "Rate limited"
            return False, f"API error ({resp.status_code})"

        return await asyncio.to_thread(_probe)
