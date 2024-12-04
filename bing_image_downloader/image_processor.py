from PIL import Image, ImageOps
import io
from typing import Optional
from .exceptions import InvalidImageError
from .config import ImageConfig

class ImageProcessor:
    def __init__(self, config: ImageConfig):
        self.config = config
        
    def process_image(self, image_data: bytes) -> Image.Image:
        if len(image_data) > self.config.max_file_size:
            raise InvalidImageError("Image file too large")
            
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Convert RGBA/LA images to RGB
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img = img.convert('RGB')
                
                # Auto-orient image based on EXIF data
                img = ImageOps.exif_transpose(img)
                
                # Validate image dimensions
                if min(img.size) < self.config.min_size:
                    raise InvalidImageError("Image too small")
                    
                # Resize if necessary
                if max(img.size) > self.config.max_size:
                    ratio = self.config.max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                return img
                
        except Exception as e:
            raise InvalidImageError(f"Failed to process image: {str(e)}") 