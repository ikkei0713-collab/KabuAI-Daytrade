"""Simple in-process event bus for decoupled component communication.

Components publish events (e.g. ``SIGNAL_GENERATED``) and other components
subscribe handlers that are called synchronously (or asynchronously if
the handler is a coroutine).

Usage::

    from core.event_bus import bus, EventType

    async def on_signal(data):
        print(f"Signal received: {data}")

    bus.subscribe(EventType.SIGNAL_GENERATED, on_signal)
    await bus.publish(EventType.SIGNAL_GENERATED, {"ticker": "7203"})
"""

from __future__ import annotations

import asyncio
import enum
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(str, enum.Enum):
    """All event types recognised by the system."""

    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    DAILY_CLOSE = "daily_close"
    STRATEGY_UPDATE = "strategy_update"


# A handler is either a plain callable or an async callable.
Handler = Callable[..., Any] | Callable[..., Coroutine[Any, Any, Any]]


class EventBus:
    """Publish / subscribe event dispatcher.

    * Handlers can be sync or async.
    * ``publish`` is always ``async`` so that async handlers are awaited.
    * Multiple handlers per event type are supported.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register *handler* to be called whenever *event_type* is published."""
        if handler in self._handlers[event_type]:
            logger.debug(
                "Handler %s already subscribed to %s — skipping",
                handler.__name__,
                event_type.value,
            )
            return
        self._handlers[event_type].append(handler)
        logger.debug(
            "Subscribed %s to %s", handler.__name__, event_type.value
        )

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a previously registered handler."""
        try:
            self._handlers[event_type].remove(handler)
            logger.debug(
                "Unsubscribed %s from %s", handler.__name__, event_type.value
            )
        except ValueError:
            logger.warning(
                "Handler %s was not subscribed to %s",
                handler.__name__,
                event_type.value,
            )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, event_type: EventType, data: Any = None) -> None:
        """Dispatch *data* to every handler registered for *event_type*.

        Handlers are invoked in subscription order.  If a handler raises,
        the exception is logged but remaining handlers still execute.
        """
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.debug("No handlers for %s", event_type.value)
            return

        logger.debug(
            "Publishing %s to %d handler(s)", event_type.value, len(handlers)
        )

        for handler in handlers:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Handler %s raised while processing %s",
                    handler.__name__,
                    event_type.value,
                )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_handlers(self, event_type: EventType) -> list[Handler]:
        """Return a copy of the handler list for *event_type*."""
        return list(self._handlers.get(event_type, []))

    def clear(self) -> None:
        """Remove all subscriptions (useful in tests)."""
        self._handlers.clear()


# Module-level singleton so components can ``from core.event_bus import bus``.
bus = EventBus()
