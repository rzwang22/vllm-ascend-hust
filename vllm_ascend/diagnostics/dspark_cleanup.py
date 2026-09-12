# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only observers of the frozen Core's shutdown escalation messages."""

import logging
import threading
from datetime import datetime, timezone

MAX_FORCE_EVENTS = 4
PROCESS_FORCE_MESSAGES = (
    "[shutdown] Process manager: force killing remaining processes count=%d",
    "[shutdown] Subprocess manager: force killing remaining processes count=%d",
)
EXECUTOR_FORCE_MESSAGES = (
    "[shutdown] Executor: workers still running after grace period; sending SIGTERM count=%d",
    "[shutdown] Executor: workers still running after SIGTERM; sending SIGKILL count=%d",
)


class ShutdownForceObserver(logging.Handler):
    """Record actual escalation requests, without changing Core or its logs."""

    def __init__(self, messages, publish):
        super().__init__(logging.WARNING)
        self.owner_thread = threading.get_ident()
        self.messages = messages
        self.events = []
        self.publish = publish

    def emit(self, record):
        if record.thread != self.owner_thread or len(self.events) >= MAX_FORCE_EVENTS:
            return
        if record.msg not in self.messages:
            return
        self.events.append({"observed_utc": datetime.now(timezone.utc).isoformat(), "message": record.getMessage()})
        self.publish(list(self.events))
