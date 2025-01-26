import glob

from PIL import Image
import os
from typing import Optional, Tuple, List
import piexif
from pathlib import Path

class ImageConverter:
    def __init__(self, quality: int = 95):
        """
        初始化圖片轉換器
        :param quality: JPG 壓縮品質 (1-100)
        """
        self.quality = quality
        self.supported_formats = {'.png', '.jpeg', '.jpg', '.webp'}

    def _get_output_path(self, input_path: str, output_dir: Optional[str] = None) -> str:
        """
        生成輸出檔案路徑
        """
        original_path = Path(input_path)
        
        if output_dir:
            output_path = Path(output_dir) / f"{original_path.stem}.jpg"
        else:
            output_path = original_path.parent / f"{original_path.stem}.jpg"
            
        # 如果檔案已存在，加上數字後綴
        counter = 1
        while output_path.exists():
            output_path = output_path.parent / f"{original_path.stem}_{counter}.jpg"
            counter += 1
            
        return str(output_path)

    def _check_image_size(self, image: Image.Image) -> Tuple[int, int]:
        """
        檢查並調整圖片尺寸以符合 Instagram 要求
        """
        width, height = image.size
        max_size = 1440
        min_size = 320

        # 檢查是否需要調整尺寸
        if width > max_size or height > max_size:
            # 等比例縮小
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return new_width, new_height
            
        elif width < min_size or height < min_size:
            # 等比例放大
            ratio = max(min_size/width, min_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return new_width, new_height
            
        return width, height

    def convert_to_jpg(self, 
                      input_path: str, 
                      output_dir: Optional[str] = None,
                      preserve_exif: bool = True) -> Optional[str]:
        """
        將圖片轉換為 JPG 格式
        
        :param input_path: 輸入圖片路徑
        :param output_dir: 輸出目錄（可選）
        :param preserve_exif: 是否保留 EXIF 資訊
        :return: 轉換後的檔案路徑，失敗則返回 None
        """
        try:
            # 檢查檔案是否存在
            if not os.path.exists(input_path):
                print(f"找不到檔案: {input_path}")
                return None

            # 檢查檔案格式
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext not in self.supported_formats:
                print(f"不支援的檔案格式: {file_ext}")
                return None

            # 開啟圖片
            with Image.open(input_path) as img:
                # 保存原始 EXIF 資料
                exif_data = None
                if preserve_exif:
                    try:
                        exif_data = piexif.load(img.info.get('exif', b''))
                    except:
                        pass

                # 轉換色彩模式
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # 檢查並調整尺寸
                new_size = self._check_image_size(img)
                if new_size != img.size:
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # 生成輸出路徑
                output_path = self._get_output_path(input_path, output_dir)

                # 儲存圖片
                save_kwargs = {
                    'quality': self.quality, 
                    'optimize': True
                }
                
                if exif_data:
                    try:
                        save_kwargs['exif'] = piexif.dump(exif_data)
                    except:
                        pass

                img.save(output_path, 'JPEG', **save_kwargs)
                print(f"圖片轉換成功: {output_path}")
                return output_path

        except Exception as e:
            print(f"轉換圖片時發生錯誤: {str(e)}")
            return None

    def batch_convert(self, 
                     input_paths: List[str], 
                     output_dir: Optional[str] = None) -> List[str]:
        """
        批次轉換多張圖片
        
        :param input_paths: 輸入圖片路徑列表
        :param output_dir: 輸出目錄（可選）
        :return: 成功轉換的檔案路徑列表
        """
        converted_paths = []
        
        for input_path in input_paths:
            result = self.convert_to_jpg(input_path, output_dir)
            if result:
                converted_paths.append(result)
                
        return converted_paths


class ImageProcessor:
    """處理圖片相關操作的類別"""
    def __init__(self, image_folder: str):
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)
            
        self.image_folder = image_folder
        
    def get_image_files(self) -> List[str]:
        """取得資料夾下所有圖片檔案"""
        types = ('*.png', '*.jpg', '*.jpeg')
        image_files = []
        for pattern in types:
            image_files.extend(glob.glob(f'{self.image_folder}/{pattern}'))
        return image_files

    def main_process(self):
        # 建立轉換器實例
        converter = ImageConverter(quality=95)
        
        # 取得所有圖片檔案
        image_files = self.get_image_files()
        
        # 轉換非jpg檔案並刪除原始檔
        converted_paths = converter.batch_convert(
            input_paths=[f for f in image_files if not f.lower().endswith('.jpg')],
            output_dir=self.image_folder
        )
        
        # 刪除已轉換的原始檔
        for path in image_files:
            if not path.lower().endswith('.jpg'):
                os.remove(path)
        
        # 更新轉換後的檔案清單
        return self.get_image_files()