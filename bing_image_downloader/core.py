import aiohttp
import hashlib
from dataclasses import dataclass
from time import time
import random


class ConnectionPool:
    def __init__(self, size: int = 10) -> None:
        self.size: int = size
        self._session: aiohttp.ClientSession | None = None
        
    async def __aenter__(self) -> aiohttp.ClientSession:
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.size),
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self._session
         
        
    async def __aexit__(self, exc_type: type[BaseException]|None, exc_val: type[BaseException]|None, exc_tb: type[BaseException]|None) -> None:
        if self._session:
            await self._session.close()
    def get_session(self) -> aiohttp.ClientSession:
        assert self._session is not None, "Session not initialized. Use within async context manager."
        return self._session


class ImageCache:
    def __init__(self, max_size: int = 1000) -> None:
        self._cached_urls: set[str] = set()
        self.max_size = max_size
        
    def _cache_impl(self, url: str) -> str:
        hash_value: str = hashlib.md5(url.encode()).hexdigest()
        return hash_value
        
    def has_image(self, url: str) -> bool:
        """Check if image URL is already cached"""
        hash_value: str = self._cache_impl(url=url)
        if hash_value in self._cached_urls:
            return True
        
        # If not in cache, add it
        if len(self._cached_urls) >= self.max_size:
            self._cached_urls.clear()  # Clear cache if full
        self._cached_urls.add(hash_value)
        return False


@dataclass
class PerformanceMetrics:
    start_time: float
    download_count: int
    failed_downloads: int
    total_bytes: int
    
class PerformanceMonitor:
    def __init__(self) -> None:
        self.metrics = PerformanceMetrics(
            start_time=time(),
            download_count=0,
            failed_downloads=0,
            total_bytes=0
        )
    
    def log_download(self, bytes_downloaded: int) -> None:
        self.metrics.download_count += 1
        self.metrics.total_bytes += bytes_downloaded
        
    def get_statistics(self) -> dict[str, float]:
        elapsed: float = time() - self.metrics.start_time
        return {
            "downloads_per_second": self.metrics.download_count / elapsed,
            "success_rate": (self.metrics.download_count - self.metrics.failed_downloads) 
                          / self.metrics.download_count,
            "total_mb_downloaded": self.metrics.total_bytes / (1024 * 1024)
        }


class CookieRotator:
    def __init__(self) -> None:
        self.user_agents: list[str] = [
            # Windows Browsers
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.2210.91",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
            
            # macOS Browsers
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
            
            # Linux Distributions
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0",
            
            # BSD Variants
            "Mozilla/5.0 (X11; FreeBSD amd64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; NetBSD amd64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; OpenBSD amd64; rv:121.0) Gecko/20100101 Firefox/121.0",
            
            # Mobile Devices
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Linux; Android 14; Samsung SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            
            # Chromium-based Browsers
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chromium/120.0.0.0 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Brave Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Vivaldi/6.5.3206.53 Chrome/120.0.0.0 Safari/537.36",
            
            # Legacy Versions (for variety)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
            
            # Additional Browsers
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Konqueror/5.0 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) QuteBrowser/2.5.4 Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/538.1 (KHTML, like Gecko) QupZilla/1.8.9 Safari/538.1",
            
            # ARM-based Systems
            "Mozilla/5.0 (Linux; aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; ARM Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
        ]
        self.current_headers: dict[str, str] = self._generate_headers()
        self.images_with_current_cookie: int = 0
        self.max_images_per_cookie: int = random.randint(a=5, b=15)

    def _generate_headers(self) -> dict[str, str]:
        user_agent: str = random.choice(seq=self.user_agents)
        return {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
            "Accept-Encoding": "none",
            "Accept-Language": "en-US,en;q=0.8",
            "Connection": "keep-alive",
        }

    def should_rotate(self) -> bool:
        return self.images_with_current_cookie >= self.max_images_per_cookie

    def rotate(self) -> dict[str, str]:
        self.current_headers = self._generate_headers()
        self.images_with_current_cookie = 0
        self.max_images_per_cookie = random.randint(a=5, b=15)
        return self.current_headers

    def increment_counter(self) -> None:
        self.images_with_current_cookie += 1