from pathlib import Path
from typing import Literal, Optional
import urllib.request
import urllib
import imghdr
import posixpath
import re
from PIL import Image
from io import BytesIO
import io
from urllib.parse import ParseResult, urlparse, quote_plus
import asyncio
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import gc
import random

from bing_image_downloader.core import ConnectionPool, ImageCache, CookieRotator
from bing_image_downloader.logger import Logger


def image_to_byte_array(image: Image.Image) -> bytes:
    """Convert PIL Image to bytes array.

    Parameters
    ----------
    image : Image.Image
        PIL Image object to convert

    Returns
    -------
    bytes
        Image converted to bytes in PNG format
    """
    imgByteArr = io.BytesIO()
    image.save(fp=imgByteArr, format="PNG")
    return imgByteArr.getvalue()


def resize(url: str, size: tuple[int, int]) -> Image.Image:
    """Download image from URL and resize to specified dimensions.

    Parameters
    ----------
    url : str
        URL of image to download and resize
    size : tuple[int, int]
        Target dimensions as (width, height) tuple

    Returns
    -------
    Image.Image
        Resized PIL Image object
    """
    response = urllib.request.urlopen(url=url)
    img: Image.Image = Image.open(fp=BytesIO(initial_bytes=response.read()))
    img = img.resize(size=size, resample=Image.Resampling.LANCZOS)  # type: ignore
    return img


class Bing:
    """Class for downloading images from Bing image search.

    Parameters
    ----------
    query : str
        Search query string
    limit : int
        Maximum number of images to download
    output_dir : Path
        Directory path to save downloaded images
    adult : bool
        Whether to enable adult content filter
    timeout : int
        Request timeout in seconds
    filter : str, optional
        Image filter type, by default ""
    resize : tuple[int, int] | None, optional
        Target dimensions (width, height) to resize images, by default None
    verbose : bool, optional
        Whether to print download progress, by default True

    Attributes
    ----------
    download_count : int
        Number of images downloaded
    query : str
        Search query string
    output_dir : Path
        Directory path to save downloaded images
    adult : bool
        Whether adult content filter is enabled
    filter : str
        Image filter type
    verbose : bool
        Whether printing download progress
    seen : set[str]
        Set of already downloaded image URLs
    limit : int
        Maximum number of images to download
    timeout : int
        Request timeout in seconds
    resize : tuple[int, int] | None
        Target dimensions for resizing images
    page_counter : int
        Counter for search result pages
    headers : dict[str, str]
        HTTP request headers
    semaphore : Semaphore
        Semaphore to limit concurrent downloads
    """

    def __init__(
        self,
        query: str,
        limit: int,
        output_dir: Path,
        adult: Literal["on", "off"],
        timeout: int,
        filter: str|None = "",
        resize: tuple[int, int] | None = None,
        verbose: bool = True,
        show_logs: bool = False,
    ) -> None:
        self.download_count = 0
        self.query: str = query
        self.output_dir: Path = output_dir
        self.adult: Literal["on", "off"] = adult
        self.filter: str|None = filter
        self.verbose: bool = verbose
        self.seen: set[str] = set()

        assert type(limit) == int, "limit must be integer"
        self.limit: int = limit
        assert type(timeout) == int, "timeout must be integer"
        self.timeout: int = timeout
        assert (type(resize) == tuple) or (
            resize is None
        ), "resize must be a tuple(height,width)"
        self.resize: tuple[int, int] | None = resize

        self.page_counter = 0

        # Add these new parameters with safe defaults
        self.min_delay = 1.0
        self.max_delay = 3.0
        self.min_cookie_rotation = 5
        self.max_cookie_rotation = 15
        self.post_rotation_min_delay = 2.0
        self.post_rotation_max_delay = 4.0

        # Initialize logger first
        self.logger = Logger(verbose=verbose, show_logs=show_logs)

        # Then initialize cookie rotator with retry mechanism
        self.cookie_rotator: Optional[CookieRotator] = None
        try:
            self.cookie_rotator = CookieRotator()
            self.headers = self.cookie_rotator.current_headers
            self.rate_limit_delay = random.uniform(self.min_delay, self.max_delay)
        except Exception as e:
            self.logger.error(f"Failed to initialize cookie rotator: {e}")
            self.rate_limit_delay = self.max_delay  # Use max delay as fallback

        # Initialize connection pool and cache
        self.connection_pool = ConnectionPool(size=16)  # Replace semaphore with connection pool
        self.image_cache = ImageCache(max_size=limit)  # Cache for downloaded images
        self.memory_manager = MemoryManager(max_memory_mb=500)  # Memory management

        self.valid_extensions: set[str] = {
            "jpe", "jpeg", "jfif", "exif", "tiff",
            "gif", "bmp", "png", "webp", "jpg"
        }

    def get_filter(
        self,
        shorthand: (
            Literal[
                "line",
                "linedrawing",
                "photo",
                "clipart",
                "gif",
                "animatedgif",
                "transparent",
            ]
            | str
        ),
    ) -> Literal[
        "",
        "+filterui:photo-linedrawing",
        "+filterui:photo-photo",
        "+filterui:photo-clipart",
        "+filterui:photo-animatedgif",
        "+filterui:photo-transparent",
    ]:
        """Convert shorthand filter name to Bing image filter URL parameter.

        Parameters
        ----------
        shorthand : Literal["line", "linedrawing", "photo", "clipart", "gif", "animatedgif", "transparent"] | str
            Shorthand name of the filter to apply. Valid values are:
            - "line"/"linedrawing": Line drawings only
            - "photo": Photographs only
            - "clipart": Clipart images only
            - "gif"/"animatedgif": Animated GIFs only
            - "transparent": Images with transparency only
            Any other value will return an empty filter string.

        Returns
        -------
        Literal["", "+filterui:photo-linedrawing", "+filterui:photo-photo", "+filterui:photo-clipart", "+filterui:photo-animatedgif", "+filterui:photo-transparent"]
            The corresponding Bing filter URL parameter, or empty string if no match.
        """
        if shorthand == "line" or shorthand == "linedrawing":
            return "+filterui:photo-linedrawing"
        elif shorthand == "photo":
            return "+filterui:photo-photo"
        elif shorthand == "clipart":
            return "+filterui:photo-clipart"
        elif shorthand == "gif" or shorthand == "animatedgif":
            return "+filterui:photo-animatedgif"
        elif shorthand == "transparent":
            return "+filterui:photo-transparent"
        else:
            return ""

    def save_image(self, link: str, file_path: Path, content: bytes | None = None) -> None:
        """Save image from URL to file path, optionally resizing."""
        if not self.resize:
            if content is None:
                request = urllib.request.Request(url=link, data=None, headers=self.headers)
                content = urllib.request.urlopen(url=request, timeout=self.timeout).read()
            if content is not None:
                if not imghdr.what(file=None, h=content):
                    print("[Error]Invalid image, not saving {}\n".format(link))
                    raise ValueError("Invalid image, not saving {}\n".format(link))
                with open(file=str(object=file_path), mode="wb") as f:
                    f.write(content)
        elif self.resize:
            img: Image.Image = resize(url=link, size=self.resize)
            image2: bytes = image_to_byte_array(image=img)
            if not imghdr.what(file=None, h=image2):
                print("[Error]Invalid image, not saving {}\n".format(link))
                raise ValueError("Invalid image, not saving {}\n".format(link))
            with open(file=str(object=file_path), mode="wb") as f:
                f.write(image2)

    @retry(
        stop=stop_after_attempt(max_attempt_number=3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(exception_types=(aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def download_with_retry(self, session: aiohttp.ClientSession, url: str) -> bytes:
        async with session.get(url=url) as response:
            response.raise_for_status()
            return await response.read()

    async def download_image_async(self, link: str, session: aiohttp.ClientSession) -> None:
        try:
            # Ensure minimum delay between requests
            await asyncio.sleep(self.rate_limit_delay)
            
            # Cookie rotation with fallback
            if self.cookie_rotator and self.cookie_rotator.should_rotate():
                try:
                    self.headers = self.cookie_rotator.rotate()
                    self.logger.debug(
                        f"Rotating cookies after {self.cookie_rotator.max_images_per_cookie} images"
                    )
                    # Randomized delay after rotation
                    rotation_delay = random.uniform(
                        self.post_rotation_min_delay,
                        self.post_rotation_max_delay
                    )
                    await asyncio.sleep(rotation_delay)
                except Exception as e:
                    self.logger.error(f"Cookie rotation failed: {e}")
                    # Fallback to longer delay without rotation
                    await asyncio.sleep(self.max_delay * 2)

            if self.image_cache.has_image(url=link):
                self.logger.debug(f"Skipping duplicate image: {link}")
                return
            
            # Add retry mechanism for failed requests
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    async with session.get(url=link, headers=self.headers) as response:
                        if response.status == 429:  # Too Many Requests
                            retry_delay = (retry_count + 1) * self.max_delay
                            self.logger.warning(
                                f"Rate limit hit, waiting {retry_delay} seconds..."
                            )
                            await asyncio.sleep(retry_delay)
                            retry_count += 1
                            continue
                            
                        response.raise_for_status()
                        if response.status == 200:
                            content: bytes = await response.read()
                            await self.memory_manager.check_memory(size=len(content))
                            
                            parsed_url: ParseResult = urlparse(url=link)
                            filename: str = posixpath.basename(p=parsed_url.path).split(sep="?")[0]
                            file_type: str = filename.split(sep=".")[-1].lower()

                            if file_type not in self.valid_extensions:
                                file_type = "jpg"

                            output_file: Path = self.output_dir.joinpath(f"Image_{self.download_count}.{file_type}")
                            self.save_image(link=link, file_path=output_file, content=content)

                            if self.cookie_rotator:
                                self.cookie_rotator.increment_counter()
                            self.download_count += 1
                            self.logger.update_stats(success=True)
                            break
                            
                except aiohttp.ClientError as e:
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        await asyncio.sleep(retry_count * self.max_delay)
                        continue
                    raise  # Re-raise if all retries failed
                    
        except Exception as e:
            self.logger.error(f"Issue getting: {link}\nError: {e}")
            self.logger.update_stats(success=False)

    async def run_async(self) -> None:
        try:
            async with self.connection_pool as session:
                consecutive_empty_pages = 0
                max_empty_pages = 3  # Stop after 3 consecutive empty pages
                
                while self.download_count < self.limit:
                    self.logger.debug(message=f"Fetching page {self.page_counter + 1}")
                    
                    # Add random delay between requests (1-3 seconds)
                    if self.page_counter > 0:
                        delay: int = 1 + (hash(str(object=self.page_counter)) % 2)
                        await asyncio.sleep(delay=delay)
                    
                    request_url: str = self._build_request_url()
                    links: list[str] = await self._fetch_image_links(request_url=request_url, session=session)
                    
                    if not links:
                        consecutive_empty_pages += 1
                        if consecutive_empty_pages >= max_empty_pages:
                            self.logger.info(message="No more images available after multiple empty pages")
                            break
                        continue
                    
                    consecutive_empty_pages = 0  # Reset counter on successful page
                    self.logger.set_total_images(total=min(self.limit, self.download_count + len(links)))
                    
                    # Process links in smaller batches
                    batch_size: int = min(5, len(links))
                    await self.process_batch(links=links, session=session, batch_size=batch_size)
                    
                    self.page_counter += 1
                    
                    if self.download_count >= self.limit:
                        break
                    
        except Exception as e:
            self.logger.error(message=f"Error during download: {str(object=e)}")
        finally:
            self.logger.info(message="\nDownload completed!")
            await self.memory_manager.cleanup()

    def run(self) -> None:
        """Run the image downloader synchronously."""
        asyncio.run(self.run_async())

    async def process_batch(self, links: list[str], session: aiohttp.ClientSession, batch_size: int = 10) -> None:
        for i in range(0, len(links), batch_size):
            batch = links[i:i + batch_size]
            tasks = [self.download_image_async(link, session) for link in batch]
            await asyncio.gather(*tasks, return_exceptions=True)  # Handle exceptions gracefully

    def _build_request_url(self) -> str:
        # Bing uses different page sizes depending on the query
        # We'll use a smaller page size to ensure more consistent results
        IMAGES_PER_PAGE: int = 25
        offset: int = self.page_counter * IMAGES_PER_PAGE
        
        # Add additional parameters that Bing uses
        params: dict[str, str] = {
            "q": self.query,
            "first": str(object=offset),
            "count": str(object=IMAGES_PER_PAGE),
            "adlt": str(object=self.adult).lower(),
            "qft": self.get_filter(shorthand=self.filter) if self.filter else "",
            "tsc": "ImageHoverTitle",  # Ensures we get image metadata
            "form": "IRFLTR",          # Image filter form
            "filters": "1",            # Enable filtering
        }
        
        # Construct query string
        query_parts: list[str] = []
        for key, value in params.items():
            if value:  # Only add non-empty parameters
                query_parts.append(f"{key}={quote_plus(string=str(object=value))}")
        
        return "https://www.bing.com/images/async?" + "&".join(query_parts)

    async def _fetch_image_links(self, request_url: str, session: aiohttp.ClientSession) -> list[str]:
        try:
            async with session.get(url=request_url, headers=self.headers) as response:
                response.raise_for_status()
                html: str = await response.text()
                
                if not html or html.isspace():
                    self.logger.warning(message="Empty response received from Bing")
                    return []
                
                # More comprehensive regex pattern to catch different URL formats
                patterns: list[str] = [
                    r'murl&quot;:&quot;(.*?)&quot;',  # Standard format
                    r'murl":"(.*?)"',                 # Alternative format
                    r'mediaurl":"(.*?)"'              # Another variation
                ]
                
                links: list[str] = []
                for pattern in patterns:
                    found_links: list[str] = re.findall(pattern=pattern, string=html)
                    links.extend(found_links)
                
                # Remove duplicates while preserving order
                links = list(dict.fromkeys(links))
                
                # Clean and validate URLs
                valid_links: list[str] = []
                for link in links:
                    cleaned_link: str = link.replace('\\', '').strip()
                    if cleaned_link.startswith(('http://', 'https://')):
                        valid_links.append(cleaned_link)
                
                if not valid_links:
                    self.logger.warning(message=f"No valid image links found on page {self.page_counter + 1}")
                else:
                    self.logger.info(message=f"Found {len(valid_links)} valid images on page {self.page_counter + 1}")
                
                return valid_links
                
        except aiohttp.ClientError as e:
            self.logger.error(message=f"Network error while fetching links: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching links: {e}")
            return []


def process_image_stream(stream: io.BytesIO, max_size: int = 1920) -> Image.Image:
    with Image.open(stream) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert(mode='RGB')
            
        # Resize if too large
        if max(img.size) > max_size:
            ratio: float = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(size=tuple(new_size), resample=Image.LANCZOS) # type: ignore
            
        return img


class MemoryManager:
    def __init__(self, max_memory_mb: int = 500) -> None:
        self.max_memory: int = max_memory_mb * 1024 * 1024
        self.current_usage = 0
        
    async def check_memory(self, size: int) -> None:
        if self.current_usage + size > self.max_memory:
            await self.cleanup()
        self.current_usage += size
        
    async def cleanup(self) -> None:
        # Force garbage collection
        gc.collect()
        self.current_usage = 0
