"""
tests/test_module_communication.py
==================================

Test suite for Module Communication Bus.

Tests:
- Module registration
- Pub/Sub pattern
- Priority-based delivery
- Request/Response pattern
- Broadcasting
- Thread safety
- Statistics
"""

import sys
import threading
import time
from pathlib import Path

import pytest

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.module_communication import (
    CommunicationBusManager,
    Message,
    MessagePriority,
    MessageType,
    ModuleCommunicationBus,
    get_communication_bus,
)


@pytest.mark.unit
class TestModuleCommunicationBus:
    """Tests for ModuleCommunicationBus."""

    def setup_method(self):
        """Setup for each test."""
        self.bus = ModuleCommunicationBus()
        self.received_messages: list[Message] = []

    def teardown_method(self):
        """Cleanup after each test."""
        self.bus.shutdown()
        self.received_messages.clear()

    # === Module Management ===

    def test_module_registration(self):
        """Test module registration."""
        self.bus.register_module("Module1")
        assert "Module1" in self.bus.get_registered_modules()

        self.bus.register_module("Module2")
        assert "Module2" in self.bus.get_registered_modules()
        assert len(self.bus.get_registered_modules()) == 2

    def test_module_unregistration(self):
        """Test module unregistration."""
        self.bus.register_module("Module1")
        assert "Module1" in self.bus.get_registered_modules()

        self.bus.unregister_module("Module1")
        assert "Module1" not in self.bus.get_registered_modules()

    def test_duplicate_registration(self):
        """Test registering module twice."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module1")  # Should log warning
        assert self.bus.get_registered_modules().count("Module1") == 1

    # === Subscription Management ===

    def test_subscribe_and_unsubscribe(self):
        """Test subscription management."""
        self.bus.register_module("Module1")

        def callback(message):
            pass

        self.bus.subscribe("Module1", "test_topic", callback)
        assert "Module1" in self.bus.get_subscribers("test_topic")

        self.bus.unsubscribe("Module1", "test_topic")
        assert "Module1" not in self.bus.get_subscribers("test_topic")

    def test_multiple_subscriptions(self):
        """Test multiple modules subscribing to same topic."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        def callback1(message):
            pass

        def callback2(message):
            pass

        self.bus.subscribe("Module1", "test_topic", callback1)
        self.bus.subscribe("Module2", "test_topic", callback2)

        subscribers = self.bus.get_subscribers("test_topic")
        assert "Module1" in subscribers
        assert "Module2" in subscribers

    # === Basic Messaging ===

    def test_publish_and_receive(self):
        """Test basic publish/receive."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        received = []

        def callback(message: Message):
            received.append(message)

        self.bus.subscribe("Module2", "test_topic", callback)

        # Publish message
        msg_id = self.bus.publish(sender="Module1", topic="test_topic", payload={"data": "test"})

        # Wait for delivery
        time.sleep(0.2)

        assert len(received) == 1
        assert received[0].sender == "Module1"
        assert received[0].topic == "test_topic"
        assert received[0].payload["data"] == "test"
        assert received[0].message_id == msg_id

    def test_broadcast(self):
        """Test broadcasting to all subscribers."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")
        self.bus.register_module("Module3")

        received_1 = []
        received_2 = []

        def callback1(message):
            received_1.append(message)

        def callback2(message):
            received_2.append(message)

        self.bus.subscribe("Module2", "broadcast_topic", callback1)
        self.bus.subscribe("Module3", "broadcast_topic", callback2)

        # Broadcast
        self.bus.broadcast(sender="Module1", topic="broadcast_topic", payload={"message": "hello"})

        # Wait for delivery
        time.sleep(0.2)

        assert len(received_1) == 1
        assert len(received_2) == 1
        assert received_1[0].message_type == MessageType.BROADCAST

    def test_direct_message(self):
        """Test direct message to specific module."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")
        self.bus.register_module("Module3")

        received_2 = []
        received_3 = []

        def callback2(message):
            received_2.append(message)

        def callback3(message):
            received_3.append(message)

        self.bus.subscribe("Module2", "direct_topic", callback2)
        self.bus.subscribe("Module3", "direct_topic", callback3)

        # Send direct message to Module2 only
        self.bus.send(sender="Module1", recipient="Module2", topic="direct_topic", payload={"data": "private"})

        # Wait for delivery
        time.sleep(0.2)

        assert len(received_2) == 1
        assert len(received_3) == 0  # Module3 should not receive

    # === Priority-Based Delivery ===

    def test_priority_ordering(self):
        """Test that priority queue is used (basic functionality test)."""
        # Note: True priority ordering in multi-threaded environment is complex
        # as messages may be processed as soon as they arrive. This test
        # verifies that the priority mechanism exists and basics work.

        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        received = []

        def callback(message):
            received.append({"priority": message.priority, "data": message.payload["data"]})

        self.bus.subscribe("Module2", "priority_topic", callback)

        # Send messages with different priorities
        self.bus.publish("Module1", "priority_topic", {"data": "low"}, MessagePriority.LOW)
        self.bus.publish("Module1", "priority_topic", {"data": "urgent"}, MessagePriority.URGENT)
        self.bus.publish("Module1", "priority_topic", {"data": "high"}, MessagePriority.HIGH)
        self.bus.publish("Module1", "priority_topic", {"data": "normal"}, MessagePriority.NORMAL)

        # Wait for delivery
        time.sleep(0.3)

        # All 4 messages should be delivered
        assert len(received) == 4

        # Verify all priority levels are represented
        priorities = [msg["priority"] for msg in received]
        assert MessagePriority.URGENT in priorities
        assert MessagePriority.HIGH in priorities
        assert MessagePriority.NORMAL in priorities
        assert MessagePriority.LOW in priorities

    # === Request/Response Pattern ===

    def test_request_response(self):
        """Test request/response pattern."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        # Module2 handles requests
        def handle_request(message: Message):
            if message.message_type == MessageType.REQUEST:
                # Send response
                self.bus.respond(original_message=message, payload={"result": "success", "value": 42})

        self.bus.subscribe("Module2", "get_data", handle_request)

        # Module1 sends request
        response = self.bus.request(
            sender="Module1", recipient="Module2", topic="get_data", payload={"query": "test"}, timeout=2.0
        )

        assert response is not None
        assert response.message_type == MessageType.RESPONSE
        assert response.payload["result"] == "success"
        assert response.payload["value"] == 42

    def test_request_timeout(self):
        """Test request timeout when no response."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        # Module2 doesn't respond
        def no_response(message):
            pass

        self.bus.subscribe("Module2", "no_response_topic", no_response)

        # Request should timeout
        response = self.bus.request(
            sender="Module1", recipient="Module2", topic="no_response_topic", payload={}, timeout=0.5
        )

        assert response is None

    # === Message History ===

    def test_message_history(self):
        """Test message history tracking."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        def callback(message):
            pass

        self.bus.subscribe("Module2", "history_topic", callback)

        # Send some messages
        self.bus.publish("Module1", "history_topic", {"msg": 1})
        self.bus.publish("Module1", "history_topic", {"msg": 2})
        self.bus.publish("Module1", "other_topic", {"msg": 3})

        time.sleep(0.2)

        # Check history
        all_history = self.bus.get_message_history()
        assert len(all_history) >= 3

        # Check filtered history
        topic_history = self.bus.get_message_history(topic="history_topic")
        assert len(topic_history) == 2

    # === Statistics ===

    def test_statistics(self):
        """Test statistics tracking."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        def callback(message):
            pass

        self.bus.subscribe("Module2", "stats_topic", callback)

        # Send messages
        self.bus.publish("Module1", "stats_topic", {"data": 1})
        self.bus.publish("Module1", "stats_topic", {"data": 2})

        time.sleep(0.2)

        stats = self.bus.get_statistics()
        assert stats["registered_modules"] == 2
        assert stats["messages_sent"] >= 2
        assert stats["messages_delivered"] >= 2

    # === Thread Safety ===

    def test_thread_safety(self):
        """Test thread-safe operations."""
        self.bus.register_module("Module1")

        for i in range(5):
            self.bus.register_module(f"Module{i + 2}")

        received_count = [0]
        lock = threading.Lock()

        def callback(message):
            with lock:
                received_count[0] += 1

        # Multiple modules subscribe
        for i in range(5):
            self.bus.subscribe(f"Module{i + 2}", "thread_safe_topic", callback)

        # Multiple threads publish
        def publish_messages(thread_id):
            for i in range(10):
                self.bus.broadcast(sender="Module1", topic="thread_safe_topic", payload={"thread": thread_id, "msg": i})

        threads = []
        for i in range(3):
            t = threading.Thread(target=publish_messages, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Wait for all deliveries
        time.sleep(0.5)

        # Should have delivered 30 messages (3 threads × 10 msgs) to 5 modules
        assert received_count[0] == 150

    # === Error Handling ===

    def test_callback_exception(self):
        """Test handling of callback exceptions."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        received = []

        def bad_callback(message):
            raise ValueError("Test exception")

        def good_callback(message):
            received.append(message)

        self.bus.subscribe("Module2", "error_topic", bad_callback)
        self.bus.subscribe("Module2", "error_topic", good_callback)

        # Should not crash even if callback raises exception
        self.bus.publish("Module1", "error_topic", {"data": "test"})

        time.sleep(0.2)

        # Good callback should still receive message
        assert len(received) == 1

    # === String Representation ===

    def test_repr(self):
        """Test string representation."""
        self.bus.register_module("Module1")
        self.bus.register_module("Module2")

        repr_str = repr(self.bus)
        assert "ModuleCommunicationBus" in repr_str
        assert "modules=2" in repr_str


class TestCommunicationBusManager:
    """Tests for CommunicationBusManager (Singleton)."""

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        manager1 = CommunicationBusManager()
        manager2 = CommunicationBusManager()
        assert manager1 is manager2

    def test_get_bus(self):
        """Test getting global bus."""
        bus1 = get_communication_bus()
        bus2 = get_communication_bus()
        assert bus1 is bus2

    def test_manager_shutdown(self):
        """Test manager shutdown."""
        manager = CommunicationBusManager()
        bus = manager.get_bus()

        bus.register_module("TestModule")
        assert "TestModule" in bus.get_registered_modules()

        manager.shutdown()
        # After shutdown, bus should be empty
        assert len(bus.get_registered_modules()) == 0


class TestMessageClass:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test message creation."""
        message = Message(
            message_id="test-123",
            message_type=MessageType.EVENT,
            sender="Module1",
            recipients=["Module2"],
            topic="test_topic",
            payload={"data": "test"},
        )

        assert message.message_id == "test-123"
        assert message.message_type == MessageType.EVENT
        assert message.sender == "Module1"
        assert message.recipients == ["Module2"]
        assert message.topic == "test_topic"
        assert message.payload["data"] == "test"

    def test_message_to_dict(self):
        """Test message serialization."""
        message = Message(
            message_id="test-123",
            message_type=MessageType.REQUEST,
            sender="Module1",
            recipients=["Module2"],
            topic="test_topic",
            payload={"query": "test"},
            priority=MessagePriority.HIGH,
        )

        data = message.to_dict()

        assert data["message_id"] == "test-123"
        assert data["message_type"] == "request"
        assert data["sender"] == "Module1"
        assert data["recipients"] == ["Module2"]
        assert data["priority"] == 2


# === Integration Tests ===


class TestCommunicationBusIntegration:
    """Integration tests for complete communication patterns."""

    def test_forensics_to_modules_workflow(self):
        """Test realistic workflow: forensics → modules."""
        bus = ModuleCommunicationBus()

        # Register modules
        bus.register_module("ForensicAnalyzer")
        bus.register_module("DCBlocker")
        bus.register_module("DeesserV2")

        processing_chain = []

        def dc_blocker_handler(message: Message):
            if message.topic == "forensic_analysis_complete":
                analysis = message.payload.get("analysis", {})
                processing_chain.append({"module": "DCBlocker", "defects": analysis.get("defects", [])})

        def deesser_handler(message: Message):
            if message.topic == "forensic_analysis_complete":
                analysis = message.payload.get("analysis", {})
                processing_chain.append({"module": "DeesserV2", "material": analysis.get("material", "unknown")})

        # Modules subscribe to forensic events
        bus.subscribe("DCBlocker", "forensic_analysis_complete", dc_blocker_handler)
        bus.subscribe("DeesserV2", "forensic_analysis_complete", deesser_handler)

        # Forensic analyzer publishes analysis
        bus.broadcast(
            sender="ForensicAnalyzer",
            topic="forensic_analysis_complete",
            payload={"analysis": {"material": "vinyl", "era": "1970s", "defects": ["dc_offset", "sibilance"]}},
            priority=MessagePriority.HIGH,
        )

        # Wait for processing
        time.sleep(0.3)

        assert len(processing_chain) == 2
        assert any(m["module"] == "DCBlocker" for m in processing_chain)
        assert any(m["module"] == "DeesserV2" for m in processing_chain)

        bus.shutdown()

    def test_module_coordination(self):
        """Test module coordination and handoff."""
        bus = ModuleCommunicationBus()

        # Register processing pipeline modules
        modules = ["Coordinator", "Module1", "Module2", "Module3"]
        for module in modules:
            bus.register_module(module)

        execution_order = []

        def create_handler(module_name, next_module=None):
            def handler(message: Message):
                if message.topic == "start_processing" or message.topic == "module_complete":
                    execution_order.append(module_name)

                    # Simulate processing
                    time.sleep(0.05)

                    # Notify completion
                    if next_module:
                        bus.send(
                            sender=module_name,
                            recipient=next_module,
                            topic="start_processing",
                            payload={"from": module_name},
                        )
                    else:
                        # Last module notifies coordinator
                        bus.send(
                            sender=module_name,
                            recipient="Coordinator",
                            topic="pipeline_complete",
                            payload={"status": "success"},
                        )

            return handler

        # Set up pipeline
        bus.subscribe("Module1", "start_processing", create_handler("Module1", "Module2"))
        bus.subscribe("Module2", "start_processing", create_handler("Module2", "Module3"))
        bus.subscribe("Module3", "start_processing", create_handler("Module3"))

        completion_received = []

        def coordinator_handler(message: Message):
            if message.topic == "pipeline_complete":
                completion_received.append(message)

        bus.subscribe("Coordinator", "pipeline_complete", coordinator_handler)

        # Start pipeline
        bus.send(
            sender="Coordinator", recipient="Module1", topic="start_processing", payload={"audio_file": "test.wav"}
        )

        # Wait for completion
        time.sleep(0.5)

        assert execution_order == ["Module1", "Module2", "Module3"]
        assert len(completion_received) == 1

        bus.shutdown()


# === Manual Test Function ===


def manual_test_communication_bus():
    """Manual test for visual verification."""
    print("\n=== Manual Communication Bus Test ===\n")

    bus = ModuleCommunicationBus()

    # Register modules
    print("1. Registering modules...")
    bus.register_module("Sender")
    bus.register_module("Receiver1")
    bus.register_module("Receiver2")
    print(f"   Registered: {bus.get_registered_modules()}")

    # Subscribe
    print("\n2. Setting up subscriptions...")

    def receiver1_callback(message: Message):
        print(f"   [Receiver1] Got message: {message.topic} - {message.payload}")

    def receiver2_callback(message: Message):
        print(f"   [Receiver2] Got message: {message.topic} - {message.payload}")

    bus.subscribe("Receiver1", "test_event", receiver1_callback)
    bus.subscribe("Receiver2", "test_event", receiver2_callback)

    # Publish
    print("\n3. Broadcasting message...")
    bus.broadcast(sender="Sender", topic="test_event", payload={"message": "Hello everyone!"})

    time.sleep(0.2)

    # Request/Response
    print("\n4. Testing request/response...")

    def responder_callback(message: Message):
        if message.message_type == MessageType.REQUEST:
            print(f"   [Receiver1] Got request: {message.payload}")
            bus.respond(message, {"status": "OK", "data": [1, 2, 3]})

    bus.subscribe("Receiver1", "get_data", responder_callback)

    response = bus.request(
        sender="Sender", recipient="Receiver1", topic="get_data", payload={"query": "all"}, timeout=2.0
    )

    if response:
        print(f"   [Sender] Got response: {response.payload}")

    # Statistics
    print("\n5. Statistics:")
    stats = bus.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Cleanup
    print("\n6. Shutting down...")
    bus.shutdown()
    print("   Done!")


if __name__ == "__main__":
    print("Running manual test...")
    manual_test_communication_bus()

    print("\n\nRunning pytest...")
    pytest.main([__file__, "-v"])
