from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Any

@dataclass
class DownloadConfig:
    max_concurrent: int
    batch_size: int
    timeout: int
    max_retries: int
    valid_extensions: set[str]

@dataclass
class ImageConfig:
    max_size: int
    min_size: int
    max_file_size: int

@dataclass
class RateLimitConfig:
    min_delay: float
    max_delay: float
    requests_per_window: int
    window_size: int

@dataclass
class MemoryConfig:
    max_memory_mb: int
    cleanup_threshold: float

class Config:
    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
            
        with open(config_path) as f:
            config: dict[str, Any] = yaml.safe_load(f)
            
        self.download = DownloadConfig(**config["download"])
        self.image = ImageConfig(**config["image"])
        self.rate_limit = RateLimitConfig(**config["rate_limiting"])
        self.memory = MemoryConfig(**config["memory"]) 