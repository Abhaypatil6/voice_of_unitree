"""
robot/ros_interface.py
----------------------
ROS 2 initialisation and node management for the G1 Voice Assistant.

Wraps rclpy to provide a clean interface that isolates ROS 2 boilerplate.
Handles node lifecycle, graceful shutdown, and provides the subscription
factory used by G1Listener.

Python 3.8 compatible.
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import Any, Callable, Optional

from logging_utils.logger import get_logger

logger = get_logger(__name__)


class ROSInterface:
    """
    Manages the rclpy node lifecycle.

    Parameters
    ----------
    node_name:
        ROS 2 node name.
    """

    def __init__(self, node_name: str = "g1_voice_assistant") -> None:
        self._node_name = node_name
        self._node: Optional[Any] = None
        self._executor: Optional[Any] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._initialised = False

    def init(self, args: Optional[list] = None) -> None:
        """Initialise rclpy and create the node."""
        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rclpy is not available.  Source your ROS 2 environment:\n"
                "    source /opt/ros/<distro>/setup.bash"
            ) from exc

        if not rclpy.ok():
            rclpy.init(args=args)

        self._rclpy = rclpy
        self._node = rclpy.create_node(self._node_name)
        self._initialised = True
        logger.info("ROS 2 node '%s' initialised", self._node_name)

    def create_subscription(
        self,
        topic: str,
        callback: Callable[[Any], None],
        qos_depth: int = 10,
    ) -> Any:
        """
        Create a ``std_msgs/msg/String`` subscription on *topic*.

        Parameters
        ----------
        topic:
            ROS topic name, e.g. ``/audio_msg``.
        callback:
            Called with each received message.
        qos_depth:
            QoS history depth.

        Returns
        -------
        rclpy Subscription object.
        """
        if not self._initialised or self._node is None:
            raise RuntimeError("ROSInterface.init() must be called first.")

        try:
            from std_msgs.msg import String  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "std_msgs is not available.  Source your ROS 2 environment."
            ) from exc

        sub = self._node.create_subscription(String, topic, callback, qos_depth)
        logger.info("Subscribed to '%s' (qos_depth=%d)", topic, qos_depth)
        return sub

    def spin_in_background(self) -> None:
        """Start spinning the node in a daemon thread."""
        if not self._initialised:
            raise RuntimeError("ROSInterface.init() must be called first.")

        def _spin() -> None:
            try:
                self._rclpy.spin(self._node)
            except Exception as exc:  # noqa: BLE001
                logger.error("ROS spin error: %s", exc)

        self._spin_thread = threading.Thread(target=_spin, daemon=True, name="ros_spin")
        self._spin_thread.start()
        logger.info("ROS 2 spin thread started")

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        """Process one batch of pending callbacks (for testing/polling)."""
        if not self._initialised:
            return
        self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def shutdown(self) -> None:
        """Destroy the node and shut down rclpy."""
        if not self._initialised:
            return
        try:
            self._node.destroy_node()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error destroying node: %s", exc)
        try:
            self._rclpy.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error shutting down rclpy: %s", exc)
        self._initialised = False
        logger.info("ROS 2 interface shut down")

    @property
    def node(self) -> Any:
        """The underlying rclpy node."""
        return self._node
