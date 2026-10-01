"""Tests for the rate limiter, in isolation from the HTTP layer.

The limiter is tested here rather than only through the API because its
behaviour at the boundary is the whole point: "the 61st request fails" is a
statement about a counter, and a test that only checks "some request eventually
fails" would pass for a limiter that fails at 5.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from semantic_search.api.middleware import FixedWindowRateLimiter


class TestFixedWindowRateLimiter:
    def test_allows_exactly_the_limit(self):
        limiter = FixedWindowRateLimiter(limit=5, window_seconds=60)
        for i in range(5):
            allowed, remaining, _ = limiter.check("client")
            assert allowed, f"request {i + 1} of 5 should be allowed"
            assert remaining == 4 - i

    def test_blocks_the_request_after_the_limit(self):
        limiter = FixedWindowRateLimiter(limit=3, window_seconds=60)
        for _ in range(3):
            limiter.check("client")
        allowed, remaining, retry_after = limiter.check("client")
        assert not allowed
        assert remaining == 0
        assert retry_after > 0

    def test_limits_are_per_client(self):
        limiter = FixedWindowRateLimiter(limit=2, window_seconds=60)
        for _ in range(2):
            limiter.check("client-a")
        assert limiter.check("client-a")[0] is False
        # One client exhausting its budget must not affect another.
        assert limiter.check("client-b")[0] is True

    def test_remaining_counts_down_monotonically(self):
        limiter = FixedWindowRateLimiter(limit=4, window_seconds=60)
        remaining = [limiter.check("c")[1] for _ in range(4)]
        assert remaining == [3, 2, 1, 0]

    def test_window_expiry_reopens_the_budget(self):
        # A one-second window makes the test fast while still exercising the
        # real eviction path rather than a mocked clock.
        limiter = FixedWindowRateLimiter(limit=2, window_seconds=1)
        limiter.check("c")
        limiter.check("c")
        assert limiter.check("c")[0] is False
        time.sleep(1.05)
        assert limiter.check("c")[0] is True

    def test_blocked_requests_do_not_extend_the_window(self):
        # If a rejected request were counted, a client polling at the limit
        # could keep itself blocked forever. The window is anchored to the
        # first request in it, not to the most recent.
        limiter = FixedWindowRateLimiter(limit=1, window_seconds=1)
        limiter.check("c")
        for _ in range(20):
            limiter.check("c")
        time.sleep(1.05)
        assert limiter.check("c")[0] is True

    def test_retry_after_shrinks_as_the_window_ages(self):
        limiter = FixedWindowRateLimiter(limit=1, window_seconds=2)
        limiter.check("c")
        first = limiter.check("c")[2]
        time.sleep(1.0)
        second = limiter.check("c")[2]
        assert second < first

    def test_retry_after_is_never_zero_when_blocked(self):
        # A Retry-After of 0 tells the client to retry immediately, which is
        # how a client ends up in a hot retry loop against a limit it cannot pass.
        limiter = FixedWindowRateLimiter(limit=1, window_seconds=1)
        limiter.check("c")
        assert limiter.check("c")[2] >= 1

    def test_reset_clears_every_bucket(self):
        limiter = FixedWindowRateLimiter(limit=1, window_seconds=60)
        limiter.check("a")
        limiter.check("b")
        limiter.reset()
        assert limiter.check("a")[0] is True
        assert limiter.check("b")[0] is True

    def test_bucket_map_is_bounded(self):
        # A long-running server must not accumulate one entry per address that
        # has ever connected. The window has to be short enough that the first
        # wave of addresses has genuinely aged out by the time the map is
        # checked -- with a 60s window, 10,050 simultaneous addresses are all
        # legitimately current and eviction is correctly refused.
        limiter = FixedWindowRateLimiter(limit=60, window_seconds=1)
        for i in range(10_050):
            limiter.check(f"client-{i}")
        time.sleep(1.05)
        limiter.check("trigger-eviction")
        assert len(limiter._hits) <= 10_000

    def test_map_keeps_current_clients_when_over_the_bound(self):
        # The complement: eviction must never drop a bucket that is still
        # inside its window, or a client over the limit would have its budget
        # silently reset and never be throttled at all.
        limiter = FixedWindowRateLimiter(limit=60, window_seconds=60)
        for i in range(10_050):
            limiter.check(f"client-{i}")
        # client-0 was tracked by the loop above, so this is a second hit on an
        # existing bucket: it must be allowed (the limit is 60) and must not
        # have been evicted and recreated.
        assert limiter.check("client-0")[0] is True
        assert len(limiter._hits) == 10_050

    def test_is_thread_safe(self):
        # The limiter is shared across a worker's threads. A counter mutated
        # without the lock can exceed its limit, and the test is a plain
        # concurrent burst rather than a mocked interleaving.
        import threading

        limiter = FixedWindowRateLimiter(limit=50, window_seconds=60)
        allowed: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(20):
                result = limiter.check("shared")[0]
                with lock:
                    allowed.append(result)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(allowed) == 50
