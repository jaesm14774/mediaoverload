import os
from pathlib import Path

import piexif
from PIL import Image


def _load_exif(image: Image.Image, preserve_exif: bool) -> dict | None:
    if not preserve_exif:
        return None
    try:
        return piexif.load(image.info.get('exif', b''))
    except Exception:  # noqa: BLE001 - malformed metadata must not block conversion.
        return None


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ('RGBA', 'LA'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        return background
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image


def _dump_exif(exif_data: dict | None) -> bytes | None:
    if not exif_data:
        return None
    try:
        return piexif.dump(exif_data)
    except Exception:  # noqa: BLE001 - malformed metadata must not block conversion.
        return None


class ImageConverter:
    def __init__(self, quality: int = 95):
        """
        初始化圖片轉換器
        :param quality: JPG 壓縮品質 (1-100)
        """
        self.quality = quality
        self.supported_formats = {'.png', '.jpeg', '.jpg', '.webp'}

    def _get_output_path(self, input_path: str, output_dir: str | None = None) -> str:
        """
        生成輸出檔案路徑
        """
        original_path = Path(input_path)
        
        if output_dir:
            output_path = Path(output_dir) / f"{original_path.stem}.jpg"
        else:
            output_path = original_path.parent / f"{original_path.stem}.jpg"
            
        counter = 1
        while output_path.exists():
            output_path = output_path.parent / f"{original_path.stem}_{counter}.jpg"
            counter += 1
            
        return str(output_path)

    def _check_image_size(self, image: Image.Image) -> tuple[int, int]:
        """
        檢查並調整圖片尺寸以符合 Instagram 要求
        """
        width, height = image.size
        max_size = 1440
        min_size = 320

        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return new_width, new_height
            
        elif width < min_size or height < min_size:
            ratio = max(min_size/width, min_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return new_width, new_height
            
        return width, height

    def convert_to_jpg(self, 
                      input_path: str, 
                      output_dir: str | None = None,
                      preserve_exif: bool = True) -> str | None:
        """
        將圖片轉換為 JPG 格式
        
        :param input_path: 輸入圖片路徑
        :param output_dir: 輸出目錄（可選）
        :param preserve_exif: 是否保留 EXIF 資訊
        :return: 轉換後的檔案路徑，失敗則返回 None
        """
        try:
            if not os.path.exists(input_path):
                print(f"找不到檔案: {input_path}")
                return None

            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext not in self.supported_formats:
                print(f"不支援的檔案格式: {file_ext}")
                return None

            with Image.open(input_path) as img:
                exif_data = _load_exif(img, preserve_exif)

                img = _to_rgb(img)

                new_size = self._check_image_size(img)
                if new_size != img.size:
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                output_path = self._get_output_path(input_path, output_dir)

                save_kwargs = {
                    'quality': self.quality, 
                    'optimize': True
                }
                
                if exif_bytes := _dump_exif(exif_data):
                    save_kwargs['exif'] = exif_bytes

                img.save(output_path, 'JPEG', **save_kwargs)
                print(f"圖片轉換成功: {output_path}")
                return output_path

        except Exception as e:  # noqa: BLE001 - the public conversion contract returns None on failure.
            print(f"轉換圖片時發生錯誤: {e!s}")
            return None
