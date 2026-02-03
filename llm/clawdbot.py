"""Clawdbot Gateway integration - sends messages through Hex instead of raw Claude."""

import logging
import asyncio
import json
import uuid
import websockets
from typing import Optional

logger = logging.getLogger(__name__)

# Gateway config - local Clawdbot instance
GATEWAY_URL = "ws://127.0.0.1:18789"
GATEWAY_TOKEN = "YOUR_GATEWAY_TOKEN_HERE"

# Protocol version
PROTOCOL_VERSION = 3


class ClawdbotClient:
    """Client that sends messages through Clawdbot Gateway (talks to Hex!)."""

    def __init__(self):
        """Initialize the Clawdbot client."""
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.request_id = 0
        self._connected = False
        self._handshake_complete = False
        self._pending_responses: dict[str, asyncio.Future] = {}
        self._chat_responses: dict[str, list[str]] = {}
        self._chat_complete_events: dict[str, asyncio.Event] = {}
        self._listener_task: Optional[asyncio.Task] = None
        logger.info("Clawdbot client initialized")

    def _next_id(self) -> str:
        """Generate next request ID."""
        self.request_id += 1
        return str(self.request_id)

    async def connect(self):
        """Connect to the Clawdbot gateway and complete handshake."""
        if self._handshake_complete:
            return

        try:
            logger.info(f"Connecting to Clawdbot gateway at {GATEWAY_URL}")
            self.ws = await websockets.connect(GATEWAY_URL)
            self._connected = True

            # Wait for connect.challenge event
            logger.info("Waiting for connect.challenge...")
            challenge_msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            challenge_data = json.loads(challenge_msg)

            if challenge_data.get("type") != "event" or challenge_data.get("event") != "connect.challenge":
                raise Exception(f"Expected connect.challenge event, got: {challenge_data}")

            logger.info("Received connect.challenge, sending connect request...")

            # Send connect request with proper format
            connect_id = self._next_id()
            connect_request = {
                "type": "req",
                "id": connect_id,
                "method": "connect",
                "params": {
                    "minProtocol": PROTOCOL_VERSION,
                    "maxProtocol": PROTOCOL_VERSION,
                    "client": {
                        "id": "gateway-client",
                        "version": "1.0.0",
                        "platform": "macos",
                        "mode": "backend"
                    },
                    "role": "operator",
                    "scopes": ["operator.read", "operator.write"],
                    "caps": [],
                    "commands": [],
                    "permissions": {},
                    "auth": {"token": GATEWAY_TOKEN},
                    "locale": "en-US",
                    "userAgent": "telegram-voice/1.0.0"
                }
            }

            await self.ws.send(json.dumps(connect_request))

            # Wait for hello-ok response
            logger.info("Waiting for hello-ok response...")
            hello_msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            hello_data = json.loads(hello_msg)

            if hello_data.get("type") != "res" or hello_data.get("id") != connect_id:
                raise Exception(f"Expected connect response, got: {hello_data}")

            if not hello_data.get("ok"):
                error = hello_data.get("error", "Unknown error")
                raise Exception(f"Connect failed: {error}")

            payload = hello_data.get("payload", {})
            if payload.get("type") != "hello-ok":
                raise Exception(f"Expected hello-ok payload, got: {payload}")

            logger.info(f"Handshake complete! Protocol: {payload.get('protocol')}")
            self._handshake_complete = True

            # Start listener task
            self._listener_task = asyncio.create_task(self._listen())

        except Exception as e:
            logger.error(f"Failed to connect to Clawdbot gateway: {e}")
            self._connected = False
            self._handshake_complete = False
            raise

    async def _listen(self):
        """Listen for messages from the gateway."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Gateway connection closed")
            self._connected = False
            self._handshake_complete = False
        except Exception as e:
            logger.error(f"Error in gateway listener: {e}")

    async def _handle_message(self, data: dict):
        """Handle incoming message from gateway."""
        msg_type = data.get("type")

        # Handle response to our request
        if msg_type == "res":
            req_id = data.get("id")
            if req_id and req_id in self._pending_responses:
                future = self._pending_responses.pop(req_id)
                if data.get("ok"):
                    future.set_result(data.get("payload"))
                else:
                    future.set_exception(Exception(data.get("error", "Unknown error")))

        # Handle events (chat streaming, etc.)
        elif msg_type == "event":
            event_name = data.get("event")
            payload = data.get("payload", {})

            if event_name == "chat":
                run_id = payload.get("runId")
                state = payload.get("state")

                if run_id:
                    # Initialize storage if needed
                    if run_id not in self._chat_responses:
                        self._chat_responses[run_id] = []

                    # Handle different states
                    if state == "delta":
                        # Streaming delta - extract text
                        message = payload.get("message")
                        if message:
                            content = message.get("content", [])
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    self._chat_responses[run_id].append(block.get("text", ""))

                    elif state == "final":
                        # Final message
                        message = payload.get("message")
                        if message:
                            content = message.get("content", [])
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        # Clear deltas and use final
                                        self._chat_responses[run_id] = [text]

                        # Signal completion
                        if run_id in self._chat_complete_events:
                            self._chat_complete_events[run_id].set()

                    elif state in ("error", "aborted"):
                        # Error or aborted
                        error_msg = payload.get("errorMessage", f"Chat {state}")
                        logger.error(f"Chat {state}: {error_msg}")
                        if run_id in self._chat_complete_events:
                            self._chat_complete_events[run_id].set()

            elif event_name == "tick":
                # Keepalive tick, ignore
                pass

            else:
                logger.debug(f"Received event: {event_name}")

    async def _send_request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Send a request to the gateway and wait for response."""
        if not self._handshake_complete:
            await self.connect()

        req_id = self._next_id()
        request = {
            "type": "req",
            "id": req_id,
            "method": method,
            "params": params
        }

        future = asyncio.Future()
        self._pending_responses[req_id] = future

        await self.ws.send(json.dumps(request))

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending_responses.pop(req_id, None)
            raise Exception(f"Request {method} timed out")

    async def get_response(self, user_message: str, session_key: str = "main") -> str:
        """
        Send a message through Clawdbot and get Hex's response.

        Args:
            user_message: The user's transcribed message
            session_key: Session to use (default: "main")

        Returns:
            Hex's response text
        """
        try:
            logger.info(f"Sending message to Hex via Clawdbot: {user_message[:100]}...")

            if not self._handshake_complete:
                await self.connect()

            # Generate idempotency key (required by protocol)
            idempotency_key = str(uuid.uuid4())

            # Set up response collection
            self._chat_responses[idempotency_key] = []
            self._chat_complete_events[idempotency_key] = asyncio.Event()

            # Prefix with voice chat context so Hex knows the source
            prefixed_message = f"[VOICE_CHAT] {user_message}"

            # Send chat.send request
            params = {
                "sessionKey": session_key,
                "message": prefixed_message,
                "idempotencyKey": idempotency_key
            }

            # Send request (don't wait for full response here, just ack)
            result = await self._send_request("chat.send", params, timeout=10.0)
            logger.info(f"Chat request acknowledged: {result}")

            # Wait for chat completion via events
            try:
                await asyncio.wait_for(
                    self._chat_complete_events[idempotency_key].wait(),
                    timeout=120.0
                )
            except asyncio.TimeoutError:
                logger.warning("Chat response timed out")

            # Collect response
            response_parts = self._chat_responses.get(idempotency_key, [])
            full_response = "".join(response_parts).strip()

            # Cleanup
            self._chat_responses.pop(idempotency_key, None)
            self._chat_complete_events.pop(idempotency_key, None)

            if not full_response:
                full_response = "I couldn't process that message. Please try again."

            logger.info(f"Hex response received: {len(full_response)} chars")
            return full_response

        except Exception as e:
            logger.error(f"Clawdbot gateway error: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the gateway."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self.ws:
            await self.ws.close()
            self._connected = False
            self._handshake_complete = False
            logger.info("Disconnected from Clawdbot gateway")
