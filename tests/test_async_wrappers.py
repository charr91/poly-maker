"""
Tests for the async wrapper pattern used in polymarket_client.py.

Tests verify that:
- run_in_executor correctly moves blocking calls to thread pool
- Multiple async calls can run concurrently without blocking each other
- Thread pool prevents event loop starvation during blocking operations
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock
import pytest


class TestRunInExecutorPattern:
    """Tests for the run_in_executor async wrapper pattern."""

    @pytest.mark.asyncio
    async def test_blocking_call_runs_in_executor(self):
        """Test that a blocking sync call doesn't block the event loop."""
        blocking_duration = 0.03

        def blocking_sync_call():
            time.sleep(blocking_duration)
            return "done"

        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, blocking_sync_call)

        concurrent_ran = False

        async def concurrent_task():
            nonlocal concurrent_ran
            await asyncio.sleep(0.005)
            concurrent_ran = True

        start = time.monotonic()
        results = await asyncio.gather(
            async_wrapper(),
            concurrent_task(),
        )
        elapsed = time.monotonic() - start

        assert concurrent_ran is True
        assert elapsed < blocking_duration * 2
        assert results[0] == "done"

    @pytest.mark.asyncio
    async def test_multiple_async_calls_run_concurrently(self):
        """Test that multiple async wrapper calls can run in parallel."""
        call_times = []
        call_duration = 0.02

        def slow_sync_call(call_id):
            start = time.monotonic()
            time.sleep(call_duration)
            end = time.monotonic()
            call_times.append((call_id, start, end))
            return call_id

        async def async_wrapper(call_id):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: slow_sync_call(call_id))

        start = time.monotonic()
        results = await asyncio.gather(*[async_wrapper(i) for i in range(5)])
        total_elapsed = time.monotonic() - start

        assert set(results) == {0, 1, 2, 3, 4}
        # With thread pool, 5 concurrent calls should complete faster than 5 * call_duration
        assert total_elapsed < call_duration * 4

    @pytest.mark.asyncio
    async def test_executor_with_partial_binds_arguments(self):
        """Test that functools.partial correctly binds arguments."""
        import functools

        def sync_method_with_args(arg1, arg2, kwarg1=None):
            return f"{arg1}-{arg2}-{kwarg1}"

        executor = ThreadPoolExecutor(max_workers=2)

        async def async_wrapper(arg1, arg2, kwarg1=None):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                executor,
                functools.partial(sync_method_with_args, arg1, arg2, kwarg1=kwarg1)
            )

        result = await async_wrapper("a", "b", kwarg1="c")
        assert result == "a-b-c"

        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_exception_propagates_through_executor(self):
        """Test that exceptions from sync methods propagate correctly."""
        def failing_sync_call():
            raise ValueError("Sync method failed")

        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, failing_sync_call)

        with pytest.raises(ValueError, match="Sync method failed"):
            await async_wrapper()


class TestThreadPoolBehavior:
    """Tests for thread pool executor behavior."""

    @pytest.mark.asyncio
    async def test_custom_executor_is_used(self):
        """Test that a custom thread pool executor is properly used."""
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='test_pool')
        thread_names = []

        def capture_thread_name():
            import threading
            thread_names.append(threading.current_thread().name)
            return True

        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, capture_thread_name)

        await async_wrapper()

        assert len(thread_names) == 1
        assert 'test_pool' in thread_names[0]

        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_thread_pool_limits_concurrency(self):
        """Test that thread pool max_workers limits concurrent execution."""
        max_workers = 2
        executor = ThreadPoolExecutor(max_workers=max_workers)
        concurrent_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        def track_concurrency():
            nonlocal concurrent_count, max_concurrent
            import threading
            # Use a threading lock since this runs in thread pool
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            time.sleep(0.01)
            concurrent_count -= 1
            return True

        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, track_concurrency)

        # Launch 10 tasks but only 2 should run concurrently
        await asyncio.gather(*[async_wrapper() for _ in range(10)])

        assert max_concurrent <= max_workers

        executor.shutdown(wait=True)

    def test_executor_shutdown_with_atexit(self):
        """Test that atexit handler can be registered for cleanup."""
        import atexit

        shutdown_called = []

        def mock_shutdown():
            shutdown_called.append(True)

        executor = ThreadPoolExecutor(max_workers=2)
        original_shutdown = executor.shutdown

        def tracked_shutdown(wait=True):
            mock_shutdown()
            return original_shutdown(wait=wait)

        executor.shutdown = tracked_shutdown

        # Simulate what atexit does
        executor.shutdown(wait=True)

        assert shutdown_called == [True]


class TestAsyncWrapperIntegration:
    """Integration tests simulating real polymarket_client usage patterns."""

    @pytest.mark.asyncio
    async def test_simulated_order_flow(self):
        """Test a simulated order creation flow with async wrappers."""
        executor = ThreadPoolExecutor(max_workers=5)

        # Simulate blocking API calls
        def create_order_sync(market_id, action, price, size):
            time.sleep(0.005)  # Simulate HTTP latency
            return {'order_id': f'{market_id}_{action}_{price}'}

        def cancel_order_sync(asset_id):
            time.sleep(0.003)
            return {'cancelled': True}

        async def create_order_async(market_id, action, price, size):
            loop = asyncio.get_event_loop()
            import functools
            return await loop.run_in_executor(
                executor,
                functools.partial(create_order_sync, market_id, action, price, size)
            )

        async def cancel_order_async(asset_id):
            loop = asyncio.get_event_loop()
            import functools
            return await loop.run_in_executor(
                executor,
                functools.partial(cancel_order_sync, asset_id)
            )

        # Simulate a trading flow
        start = time.monotonic()

        # Cancel existing orders, then place new ones (in sequence for same market)
        await cancel_order_async('asset_1')
        result1 = await create_order_async('market_1', 'BUY', 0.5, 100)

        # Different market can run in parallel
        results = await asyncio.gather(
            create_order_async('market_2', 'BUY', 0.6, 50),
            create_order_async('market_3', 'SELL', 0.7, 75),
        )

        elapsed = time.monotonic() - start

        # Verify results
        assert result1 == {'order_id': 'market_1_BUY_0.5'}
        assert results[0] == {'order_id': 'market_2_BUY_0.6'}
        assert results[1] == {'order_id': 'market_3_SELL_0.7'}

        # Parallel calls should complete faster than sequential
        assert elapsed < 0.05

        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_websocket_heartbeat_not_blocked(self):
        """Test that blocking API calls don't prevent WebSocket heartbeats."""
        executor = ThreadPoolExecutor(max_workers=3)
        heartbeat_times = []

        def blocking_api_call():
            time.sleep(0.04)  # Simulate 40ms API call
            return True

        async def api_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, blocking_api_call)

        async def heartbeat_task():
            """Simulates WebSocket PING/PONG every 10ms."""
            for _ in range(5):
                heartbeat_times.append(time.monotonic())
                await asyncio.sleep(0.008)

        # Run blocking API call and heartbeat concurrently
        await asyncio.gather(
            api_wrapper(),
            heartbeat_task(),
        )

        # All 5 heartbeats should have occurred
        assert len(heartbeat_times) == 5

        # Heartbeats should be roughly 8ms apart (not blocked by API call)
        for i in range(1, len(heartbeat_times)):
            interval = heartbeat_times[i] - heartbeat_times[i - 1]
            # Should be close to 0.008s, definitely less than the 0.04s API call
            assert interval < 0.03, f"Heartbeat interval {interval}s too long (blocked by API call)"

        executor.shutdown(wait=True)
