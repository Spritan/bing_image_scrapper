class BingImageError(Exception):
    """Base exception for Bing Image Downloader"""
    pass

class DownloadError(BingImageError):
    """Raised when image download fails"""
    pass

class RateLimitError(BingImageError):
    """Raised when hitting rate limits"""
    pass

class InvalidImageError(BingImageError):
    """Raised when downloaded content is not a valid image"""
    pass 