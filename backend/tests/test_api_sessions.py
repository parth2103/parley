"""Tests for server-side call cleanup; no provider clients are created."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.api.main import disconnect_all_sessions


def test_disconnect_all_sessions_removes_orphaned_call():
    connection = SimpleNamespace(disconnect=AsyncMock())
    worker = SimpleNamespace(cancel=AsyncMock())

    async def exercise():
        job = asyncio.create_task(asyncio.sleep(0))
        app = SimpleNamespace(
            state=SimpleNamespace(sessions={"orphan": (connection, worker, job)})
        )
        await disconnect_all_sessions(app)
        assert app.state.sessions == {}

    asyncio.run(exercise())
    worker.cancel.assert_awaited_once()
    connection.disconnect.assert_awaited_once()
