"""Claude API integration for LLM responses."""

import logging
from anthropic import Anthropic
import config

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Claude API client for generating responses."""

    def __init__(self):
        """Initialize the Claude API client."""
        self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.conversation_history = []
        logger.info("Claude API client initialized")

    def get_response(self, user_message: str) -> str:
        """
        Get a response from Claude for the user's message.

        Args:
            user_message: The user's transcribed message

        Returns:
            Claude's response text
        """
        try:
            logger.info(f"Sending message to Claude: {user_message[:100]}...")

            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Get response from Claude
            response = self.client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=1024,
                system="You are a helpful voice assistant. Keep your responses concise and conversational, as they will be spoken aloud.",
                messages=self.conversation_history
            )

            # Extract assistant's response
            assistant_message = response.content[0].text

            # Add assistant message to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # Keep conversation history manageable (last 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            logger.info(f"Claude response received: {len(assistant_message)} chars")
            return assistant_message

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
