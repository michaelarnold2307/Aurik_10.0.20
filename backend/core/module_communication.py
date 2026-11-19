"""
core/module_communication.py
Module Communication Bus
========================

Event-based Inter-Module Communication System für AURIK.
Ermöglicht:
- Publish/Subscribe Pattern
- Event Broadcasting
- Request/Response Pattern
- Module Discovery
- Message Routing
- Priority Queues

Version: 1.0.0
Author: AURIK Team
Date: 10. Februar 2026
"""

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessagePriority(Enum):
    """Message priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class MessageType(Enum):
    """Types of messages."""

    EVENT = "event"  # Regular event notification
    REQUEST = "request"  # Request for data/action
    RESPONSE = "response"  # Response to request
    COMMAND = "command"  # Direct command to module
    BROADCAST = "broadcast"  # Broadcast to all modules


@dataclass
class Message:
    """
    Message object for inter-module communication.
    """

    message_id: str
    message_type: MessageType
    sender: str
    recipients: list[str]  # Empty list = broadcast
    topic: str
    payload: dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None  # For request/response matching
    reply_to: str | None = None  # For responses

    def to_dict(self) -> dict[str, Any]:
        """Konvertiert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "recipients": self.recipients,
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
        }


class ModuleCommunicationBus:
    """
    Central Communication Bus for AURIK modules.

    Features:
    - Publish/Subscribe Pattern
    - Event Broadcasting
    - Request/Response Pattern
    - Priority-based Message Delivery
    - Topic-based Routing
    - Module Registration & Discovery
    - Message History
    - Thread-safe Operations

    Usage:
        # Initialize bus
        bus = ModuleCommunicationBus()

        # Register module
        bus.register_module("Module1")

        # Subscribe to topics
        bus.subscribe("Module1", "forensic_analysis_complete", callback_function)

        # Publish event
        bus.publish(
            sender="Module1",
            topic="processing_started",
            payload={"audio_length": 1204}
        )

        # Request/Response
        response = bus.request(
            sender="Module1",
            recipient="ForensicAnalyzer",
            topic="get_analysis",
            payload={"session_id": "123"},
            timeout=5.0
        )
    """

    VERSION = "1.0.0"

    def __init__(self, max_history_size: int = 1000):
        """
        Initialisiert communication bus.

        Args:
            max_history_size: Maximum number of messages to keep in history
        """
        # Thread-safe state
        self._lock = threading.RLock()

        # Module registry
        self._modules: set[str] = set()

        # Subscriptions: topic → {module: [callbacks]}
        self._subscriptions: dict[str, dict[str, list[Callable]]] = {}

        # Message queues: module → PriorityQueue
        self._message_queues: dict[str, queue.PriorityQueue] = {}

        # Request/Response tracking
        self._pending_requests: dict[str, threading.Event] = {}
        self._responses: dict[str, Message] = {}

        # Message history
        self._message_history: list[Message] = []
        self._max_history_size = max_history_size

        # Worker threads
        self._workers: dict[str, threading.Thread] = {}
        self._running = False

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
            "requests_sent": 0,
            "responses_sent": 0,
        }

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModuleCommunicationBus initialized")

    # === Module Management ===

    def register_module(self, module_name: str) -> None:
        """
        Registriert a module with the bus.

        Args:
            module_name: Name of the module
        """
        with self._lock:
            if module_name in self._modules:
                self.logger.warning("Module already registered: %s", module_name)
                return

            self._modules.add(module_name)
            self._message_queues[module_name] = queue.PriorityQueue()

            # Start worker thread for this module
            if not self._running:
                self._running = True

            worker = threading.Thread(target=self._process_messages, args=(module_name,), daemon=True)
            worker.start()
            self._workers[module_name] = worker

            self.logger.info("Module registered: %s", module_name)

    def unregister_module(self, module_name: str) -> None:
        """
        Unregister a module from the bus.

        Args:
            module_name: Name of the module
        """
        with self._lock:
            if module_name not in self._modules:
                return

            self._modules.remove(module_name)

            # Clean up subscriptions
            for topic in list(self._subscriptions.keys()):
                if module_name in self._subscriptions[topic]:
                    del self._subscriptions[topic][module_name]
                    if not self._subscriptions[topic]:
                        del self._subscriptions[topic]

            # Clean up message queue
            if module_name in self._message_queues:
                del self._message_queues[module_name]

            self.logger.info("Module unregistered: %s", module_name)

    def get_registered_modules(self) -> list[str]:
        """
        Gibt zurück: list of registered modules.

        Returns:
            List of module names
        """
        with self._lock:
            return list(self._modules)

    # === Subscription Management ===

    def subscribe(self, module_name: str, topic: str, callback: Callable[[Message], None]) -> None:
        """
        Abonniert a module to a topic.

        Args:
            module_name: Name of the subscribing module
            topic: Topic to subscribe to
            callback: Callback function (receives Message object)
        """
        with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = {}

            if module_name not in self._subscriptions[topic]:
                self._subscriptions[topic][module_name] = []

            self._subscriptions[topic][module_name].append(callback)

            self.logger.debug("%s subscribed to '%s'", module_name, topic)

    def unsubscribe(self, module_name: str, topic: str, callback: Callable | None = None) -> None:
        """
        Unsubscribe a module from a topic.

        Args:
            module_name: Name of the module
            topic: Topic to unsubscribe from
            callback: Optional specific callback to remove
        """
        with self._lock:
            if topic not in self._subscriptions:
                return

            if module_name not in self._subscriptions[topic]:
                return

            if callback is None:
                # Remove all callbacks for this module
                del self._subscriptions[topic][module_name]
            else:
                # Remove specific callback
                if callback in self._subscriptions[topic][module_name]:
                    self._subscriptions[topic][module_name].remove(callback)

            # Clean up empty subscriptions
            if module_name in self._subscriptions[topic] and not self._subscriptions[topic][module_name]:
                del self._subscriptions[topic][module_name]
            if not self._subscriptions[topic]:
                del self._subscriptions[topic]

            self.logger.debug("%s unsubscribed from '%s'", module_name, topic)

    def get_subscribers(self, topic: str) -> list[str]:
        """
        Gibt zurück: list of modules subscribed to a topic.

        Args:
            topic: Topic name

        Returns:
            List of module names
        """
        with self._lock:
            if topic not in self._subscriptions:
                return []
            return list(self._subscriptions[topic].keys())

    # === Message Publishing ===

    def publish(
        self,
        sender: str,
        topic: str,
        payload: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        recipients: list[str] | None = None,
    ) -> str:
        """
        Publish a message to a topic.

        Args:
            sender: Name of sending module
            topic: Message topic
            payload: Message payload
            priority: Message priority
            recipients: Optional list of specific recipients (None = broadcast)

        Returns:
            Message ID
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.EVENT if recipients else MessageType.BROADCAST,
            sender=sender,
            recipients=recipients or [],
            topic=topic,
            payload=payload,
            priority=priority,
        )

        self._send_message(message)

        with self._lock:
            self._stats["messages_sent"] += 1

        return message.message_id

    def broadcast(
        self, sender: str, topic: str, payload: dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """
        Broadcast a message to all subscribed modules.

        Args:
            sender: Name of sending module
            topic: Message topic
            payload: Message payload
            priority: Message priority

        Returns:
            Message ID
        """
        return self.publish(sender, topic, payload, priority=priority, recipients=None)

    def send(
        self,
        sender: str,
        recipient: str,
        topic: str,
        payload: dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """
        Sendet a direct message to a specific module.

        Args:
            sender: Name of sending module
            recipient: Name of receiving module
            topic: Message topic
            payload: Message payload
            priority: Message priority

        Returns:
            Message ID
        """
        return self.publish(sender, topic, payload, priority=priority, recipients=[recipient])

    # === Request/Response Pattern ===

    def request(
        self, sender: str, recipient: str, topic: str, payload: dict[str, Any], timeout: float = 5.0
    ) -> Message | None:
        """
        Sendet a request and wait for response.

        Args:
            sender: Name of sending module
            recipient: Name of receiving module
            topic: Request topic
            payload: Request payload
            timeout: Timeout in seconds

        Returns:
            Response Message or None if timeout
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.REQUEST,
            sender=sender,
            recipients=[recipient],
            topic=topic,
            payload=payload,
            reply_to=sender,
        )

        # Create event for waiting
        response_event = threading.Event()
        correlation_id = message.message_id

        with self._lock:
            self._pending_requests[correlation_id] = response_event
            self._stats["requests_sent"] += 1

        # Send request
        self._send_message(message)

        # Wait for response
        if response_event.wait(timeout):
            with self._lock:
                response = self._responses.pop(correlation_id, None)
                del self._pending_requests[correlation_id]
            return response
        else:
            # Timeout
            with self._lock:
                if correlation_id in self._pending_requests:
                    del self._pending_requests[correlation_id]
            self.logger.warning("Request timeout: %s → %s (%s)", sender, recipient, topic)
            return None

    def respond(self, original_message: Message, payload: dict[str, Any]) -> str:
        """
        Sendet a response to a request.

        Args:
            original_message: Original request message
            payload: Response payload

        Returns:
            Response message ID
        """
        response = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender=original_message.recipients[0] if original_message.recipients else "unknown",
            recipients=[original_message.sender],
            topic=f"{original_message.topic}_response",
            payload=payload,
            correlation_id=original_message.message_id,
        )

        self._send_message(response)

        # Signal waiting request
        with self._lock:
            if original_message.message_id in self._pending_requests:
                self._responses[original_message.message_id] = response
                self._pending_requests[original_message.message_id].set()
            self._stats["responses_sent"] += 1

        return response.message_id

    # === Internal Message Processing ===

    def _send_message(self, message: Message) -> None:
        """
        Sendet message to appropriate queues.

        Args:
            message: Message to send
        """
        with self._lock:
            # Add to history
            self._message_history.append(message)
            if len(self._message_history) > self._max_history_size:
                self._message_history.pop(0)

            # Determine recipients
            if message.recipients:
                # Direct message
                target_modules = [m for m in message.recipients if m in self._modules]
            else:
                # Broadcast to subscribers
                target_modules = self.get_subscribers(message.topic)

            # Add to each recipient's queue
            for module in target_modules:
                if module in self._message_queues:
                    # Priority queue: (priority, timestamp, message)
                    self._message_queues[module].put(
                        (-message.priority.value, message.timestamp, message)  # Negative for correct priority order
                    )

    def _process_messages(self, module_name: str) -> None:
        """
        Worker thread to process messages for a module.

        Args:
            module_name: Name of the module
        """
        while self._running:
            try:
                # Get message from queue (timeout for shutdown)
                _, _, message = self._message_queues[module_name].get(timeout=0.1)

                # Deliver message to subscribers
                self._deliver_message(module_name, message)

                with self._lock:
                    self._stats["messages_delivered"] += 1

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("Message processing error (%s): %s", module_name, e)

    def _deliver_message(self, module_name: str, message: Message) -> None:
        """
        Deliver message to module's callbacks.

        Args:
            module_name: Name of the module
            message: Message to deliver
        """
        with self._lock:
            if message.topic not in self._subscriptions:
                return

            if module_name not in self._subscriptions[message.topic]:
                return

            callbacks = self._subscriptions[message.topic][module_name].copy()

        # Execute callbacks (outside lock to avoid deadlock)
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error("Callback error (%s, %s): %s", module_name, message.topic, e)

    # === Statistics & Monitoring ===

    def get_statistics(self) -> dict[str, Any]:
        """
        Gibt zurück: bus statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "registered_modules": len(self._modules),
                "subscriptions": len(self._subscriptions),
                "messages_sent": self._stats["messages_sent"],
                "messages_delivered": self._stats["messages_delivered"],
                "messages_dropped": self._stats["messages_dropped"],
                "requests_sent": self._stats["requests_sent"],
                "responses_sent": self._stats["responses_sent"],
                "pending_requests": len(self._pending_requests),
                "history_size": len(self._message_history),
            }

    def get_message_history(self, topic: str | None = None, limit: int = 100) -> list[Message]:
        """
        Gibt zurück: message history.

        Args:
            topic: Optional topic filter
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        with self._lock:
            messages = self._message_history[-limit:]

            if topic:
                messages = [m for m in messages if m.topic == topic]

            return messages

    # === Lifecycle ===

    def shutdown(self) -> None:
        """Shutdown the communication bus."""
        self.logger.info("Shutting down communication bus...")
        self._running = False

        # Wait for workers to finish (with timeout)
        for worker in self._workers.values():
            worker.join(timeout=1.0)

        with self._lock:
            self._modules.clear()
            self._subscriptions.clear()
            self._message_queues.clear()
            self._pending_requests.clear()
            self._responses.clear()

        self.logger.info("Communication bus shut down")

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            "ModuleCommunicationBus("
            f"modules={stats['registered_modules']}, "
            f"subscriptions={stats['subscriptions']}, "
            f"messages_sent={stats['messages_sent']})"
        )


# === Global Bus Manager ===


class CommunicationBusManager:
    """
    Global Communication Bus Manager (Singleton).
    """

    _instance = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:  # pylint: disable=access-member-before-definition
            return

        self._bus = ModuleCommunicationBus()
        self._initialized = True

        self.logger = logging.getLogger(__name__)

    def get_bus(self) -> ModuleCommunicationBus:
        """Gibt zurück: the global communication bus."""
        return self._bus

    def shutdown(self) -> None:
        """Shutdown the global bus."""
        self._bus.shutdown()


# === Convenience Functions ===


def get_communication_bus() -> ModuleCommunicationBus:
    """Gibt zurück: global communication bus (singleton)."""
    return CommunicationBusManager().get_bus()
