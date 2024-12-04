import sys
import shutil
from pathlib import Path
from typing import Literal

from .bing import Bing


def download(
    query: str,
    limit: int = 100,
    output_dir: str = "dataset",
    adult_filter_off: bool = True,
    force_replace: bool = False,
    timeout: int = 60,
    filter: str = "",
    resize: tuple[int, int] | None = None,
    verbose: bool = True,
    show_logs: bool = False,
) -> None:
    """Download images from Bing image search.

    Parameters
    ----------
    query : str
        Search query string
    limit : int, optional
        Maximum number of images to download, by default 100
    output_dir : str, optional
        Base directory to save downloaded images, by default 'dataset'
    adult_filter_off : bool, optional
        Whether to disable adult content filter, by default True
    force_replace : bool, optional
        Whether to delete and recreate existing output directory, by default False
    timeout : int, optional
        Request timeout in seconds, by default 60
    filter : str, optional
        Image filter type (e.g. "photo", "clipart"), by default ""
    resize : tuple[int, int] | None, optional
        Target dimensions (width, height) to resize images, by default None
    verbose : bool, optional
        Whether to print download progress, by default True
    show_logs : bool, optional
        Whether to show detailed logs, by default False

    Raises
    ------
    SystemExit
        If output directory creation fails
    """
    adult: Literal["on", "off"] = "on" if not adult_filter_off else "off"

    image_dir: Path = Path(output_dir).joinpath(query).absolute()

    if force_replace:
        if Path.is_dir(self=image_dir):
            shutil.rmtree(image_dir)

    # check directory and create if necessary
    try:
        if not Path.is_dir(self=image_dir):
            Path.mkdir(self=image_dir, parents=True)

    except Exception as e:
        print("[Error]Failed to create directory.", e)
        sys.exit(1)

    print("[%] Downloading Images to {}".format(str(object=image_dir.absolute())))
    bing: Bing = Bing(
        query=query,
        limit=limit,
        output_dir=image_dir,
        adult=adult,
        timeout=timeout,
        filter=filter,
        resize=resize,
        verbose=verbose,
    )
    bing.run()


if __name__ == "__main__":
    download(query="dog", output_dir="..\\Users\\cat", limit=10, timeout=1)
