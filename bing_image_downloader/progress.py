from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn
)
from rich.console import Console
from dataclasses import dataclass
from typing import Optional

@dataclass
class DownloadProgress:
    total: int
    completed: int = 0
    failed: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0

class ProgressManager:
    def __init__(self, total: int):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        self.stats = DownloadProgress(total=total)
        self._task_id: Optional[int] = None
        
    def __enter__(self) -> 'ProgressManager':
        self.progress.start()
        self._task_id = self.progress.add_task(
            "[cyan]Downloading images...",
            total=self.stats.total
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.progress.stop()
        
    def update(self, success: bool = True) -> None:
        if success:
            self.stats.completed += 1
        else:
            self.stats.failed += 1
            
        if self._task_id is not None:
            self.progress.update(
                self._task_id,
                advance=1,
                description=f"[cyan]Downloaded: {self.stats.completed} Failed: {self.stats.failed}"
            ) 