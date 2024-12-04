import asyncio
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Deque

@dataclass
class RateLimiter:
    requests_per_window: int
    window_size: float
    min_delay: float
    max_delay: float
    request_times: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    
    async def acquire(self) -> None:
        now = time.time()
        
        # Remove old requests outside the window
        while self.request_times and self.request_times[0] < now - self.window_size:
            self.request_times.popleft()
            
        # If we've hit the limit, wait until we can make another request
        if len(self.request_times) >= self.requests_per_window:
            wait_time = self.request_times[0] + self.window_size - now
            await asyncio.sleep(max(wait_time, self.min_delay))
            
        # Add random delay within bounds
        delay = self.min_delay + (self.max_delay - self.min_delay) * (len(self.request_times) / self.requests_per_window)
        await asyncio.sleep(delay)
        
        self.request_times.append(now) 