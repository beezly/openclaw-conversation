"""Conversation agent for OpenClaw."""

from __future__ import annotations

import logging
from typing import Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, device_registry as dr, intent
from homeassistant.util import ulid

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_VERIFY_SSL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_VERIFY_SSL,
)

_LOGGER = logging.getLogger(__name__)


class OpenClawConversationAgent(conversation.AbstractConversationAgent):
    """OpenClaw conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self._base_url = entry.data[CONF_BASE_URL]
        self._api_key = entry.data[CONF_API_KEY]
        self._model = entry.data.get(CONF_MODEL, DEFAULT_MODEL)
        self._timeout = entry.data.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
        self._verify_ssl = entry.data.get(CONF_VERIFY_SSL, DEFAULT_VERIFY_SSL)
        self._conversations: dict[str, list[dict]] = {}

    @property
    def attribution(self) -> dict[str, str]:
        """Return attribution."""
        return {"name": "Powered by OpenClaw", "url": "https://openclaw.ai"}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return "*"

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        conversation_id = user_input.conversation_id or ulid.ulid_now()

        # Build system prompt
        system_content = "You are a helpful home assistant."

        if user_input.device_id:
            device = dr.async_get(self.hass).async_get(user_input.device_id)
            if device and device.area_id:
                area = ar.async_get(self.hass).async_get_area(device.area_id)
                if area:
                    system_content += f" The request came from the {area.name}."

        if user_input.extra_system_prompt:
            system_content += f"\n{user_input.extra_system_prompt}"

        # Get or create conversation history
        messages = self._conversations.get(conversation_id, [])

        # Add user message
        messages.append({"role": "user", "content": user_input.text})

        # Call OpenClaw with system prompt prepended
        try:
            full_messages = [{"role": "system", "content": system_content}] + messages
            response_text = await self._call_openclaw(full_messages)
        except Exception as err:
            _LOGGER.error("Error calling OpenClaw: %s", err)
            response_text = "Erreur de communication avec OpenClaw."

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response_text})

        # Keep conversation history (limit to last 20 messages)
        self._conversations[conversation_id] = messages[-20:]

        # Build response
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)

        return conversation.ConversationResult(
            response=response,
            conversation_id=conversation_id,
        )

    async def _call_openclaw(self, messages: list[dict]) -> str:
        """Call OpenClaw chat completions API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": messages,
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                ssl=None if self._verify_ssl else False,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"OpenClaw returned {resp.status}: {body[:200]}"
                    )

                data = await resp.json()
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No response from OpenClaw")

                return choices[0]["message"]["content"]
