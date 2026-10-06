"""Pace every request (including retries) against QPM and reserved token budgets."""

import asyncio
import time
from math import isfinite


class RequestPacer:
    def __init__(self, qpm=None, tpm=None, tokens_per_request=20000):
        for limit in (qpm, tpm, tokens_per_request):
            if limit is not None and (not isfinite(limit) or limit <= 0):
                raise ValueError(
                    "Rate limits and token reservation must be positive and finite"
                )
        if tpm is not None and tokens_per_request > tpm:
            raise ValueError(
                "tokens_per_request cannot exceed the one-minute token budget"
            )
        intervals = [60 / qpm] if qpm else []
        if tpm:
            intervals.append(60 * tokens_per_request / tpm)
        self.interval = max(intervals, default=0)
        self.tokens_per_request = tokens_per_request
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        if not self.interval:
            return
        # Hold the lock through the wait: delayed tasks cannot catch up in a burst.
        async with self.lock:
            delay = self.next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = time.monotonic() + self.interval

    def observe(self, usage):
        total = (usage or {}).get("total_tokens")
        if isinstance(total, (int, float)) and total > self.tokens_per_request:
            # A reservation underestimate must be surfaced, not silently treated as safe.
            raise ValueError(
                f"Response used {total} tokens, exceeding tokens_per_request="
                f"{self.tokens_per_request}; increase reservation before continuing"
            )
