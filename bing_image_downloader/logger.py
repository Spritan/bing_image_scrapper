from dataclasses import dataclass
from time import time

@dataclass
class DownloadStats:
    total_found: int = 0
    successful: int = 0
    failed: int = 0
    start_time: float = time()
    
    def get_summary(self) -> str:
        elapsed = time() - self.start_time
        return (
            f"\rProgress: [{self.successful + self.failed} "
            f"✓{self.successful} ✗{self.failed}] "
            f"({elapsed:.1f}s)"
        )

class Logger:
    def __init__(self, verbose: bool = True, show_logs: bool = False):
        self.verbose = verbose
        self.show_logs = show_logs
        self.stats = DownloadStats()
        
    def update_stats(self, success: bool = True) -> None:
        if success:
            self.stats.successful += 1
        else:
            self.stats.failed += 1
        if self.verbose:
            print(self.stats.get_summary(), end="", flush=True)
            
    def set_total_images(self, total: int) -> None:
        self.stats.total_found = total
        if self.verbose and self.show_logs:
            print(f"\nFound {total} images")
            
    def info(self, message: str) -> None:
        if self.verbose and self.show_logs:
            print(f"\n[INFO] {message}")
                
    def error(self, message: str) -> None:
        if self.show_logs:
            print(f"\n[ERROR] {message}")
            
    def warning(self, message: str) -> None:
        if self.verbose and self.show_logs:
            print(f"\n[WARNING] {message}")
            
    def debug(self, message: str) -> None:
        if self.verbose and self.show_logs:
            print(f"\n[DEBUG] {message}")