import argparse
from bing_image_downloader import downloader

def main() -> None:
    parser = argparse.ArgumentParser(description='Download images from Bing')
    parser.add_argument('query', help='Search query string')
    parser.add_argument('--filter', default="", help='Image filter type')
    parser.add_argument('--logs', action='store_true', help='Show detailed logs')
    
    args: argparse.Namespace = parser.parse_args()
    
    downloader.download(
        query=args.query,
        limit=10,
        output_dir="dataset",
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        filter=args.filter,
        verbose=True,
        show_logs=args.logs
    )

if __name__ == "__main__":
    main()

