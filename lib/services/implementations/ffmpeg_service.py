"""FFmpeg 服務"""
import subprocess
import os
import tempfile
from typing import List, Optional, Literal
from pathlib import Path
import logging


class FFmpegService:
    """FFmpeg 服務
    
    提供影片處理功能：提取幀、合併影片、合併音訊、格式轉換等。
    """
    
    def __init__(self):
        """初始化 FFmpeg 服務並檢查 ffmpeg 是否可用"""
        self.logger = logging.getLogger(__name__)
        self._check_ffmpeg_installed()
    
    def _check_ffmpeg_installed(self):
        """檢查 ffmpeg 是否已安裝"""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            self.logger.info("FFmpeg 可用")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "FFmpeg is not installed or not in PATH. "
                "Please install ffmpeg: https://ffmpeg.org/download.html"
            ) from e
    
    def extract_last_frame(
        self,
        video_path: str,
        output_path: str
    ) -> str:
        """提取影片最後一幀
        
        用於長影片生成，每個片段使用前一片段的最後一幀作為條件。
        
        Args:
            video_path: 輸入影片路徑
            output_path: 輸出圖片路徑
            
        Returns:
            提取的圖片路徑
            
        Raises:
            RuntimeError: 提取失敗時
        """
        try:
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                video_path
            ]
            
            result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            total_frames = int(result.stdout.strip())
            last_frame_idx = total_frames - 1
            
            extract_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'select=eq(n\\,{last_frame_idx})',
                '-vframes', '1',
                '-y',
                output_path
            ]
            
            subprocess.run(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            self.logger.info(f"已提取最後一幀: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to extract last frame from {video_path}: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def concat_videos(
        self,
        video_paths: List[str],
        output_path: str,
        method: Literal['demuxer', 'filter'] = 'demuxer'
    ) -> str:
        """合併多個影片
        
        Args:
            video_paths: 要合併的影片路徑列表（按順序）
            output_path: 輸出影片路徑
            method: 合併方法
                - 'demuxer': 快速，無需重新編碼（需要相同編碼）
                - 'filter': 較慢但可處理不同格式
                
        Returns:
            合併後的影片路徑
            
        Raises:
            RuntimeError: 合併失敗時
        """
        if not video_paths:
            raise ValueError("video_paths cannot be empty")
        
        if len(video_paths) == 1:
            import shutil
            shutil.copy2(video_paths[0], output_path)
            return output_path
        
        if method == 'demuxer':
            return self._concat_demuxer(video_paths, output_path)
        elif method == 'filter':
            return self._concat_filter(video_paths, output_path)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _concat_demuxer(self, video_paths: List[str], output_path: str) -> str:
        """使用 concat demuxer 合併影片"""
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            filelist_path = f.name
            for video_path in video_paths:
                abs_path = os.path.abspath(video_path)
                f.write(f"file '{abs_path}'\n")
        
        try:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to concatenate videos: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            # Clean up temp file
            try:
                os.unlink(filelist_path)
            except:
                pass
    
    def _concat_filter(self, video_paths: List[str], output_path: str) -> str:
        """使用 concat filter 合併影片"""
        inputs = []
        for path in video_paths:
            inputs.extend(['-i', path])
        
        filter_parts = []
        for i in range(len(video_paths)):
            filter_parts.append(f'[{i}:v][{i}:a]')
        
        filter_complex = (
            f"{''.join(filter_parts)}concat=n={len(video_paths)}:v=1:a=1[v][a]"
        )
        
        cmd = [
            'ffmpeg',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', '[a]',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to concatenate videos with filter: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        audio_codec: str = 'aac'
    ) -> str:
        """合併音訊與影片
        
        如果影片已有音訊，將被新音訊取代。
        
        Args:
            video_path: 輸入影片路徑
            audio_path: 輸入音訊路徑
            output_path: 輸出影片路徑
            audio_codec: 音訊編碼（預設: 'aac'）
            
        Returns:
            輸出影片路徑
            
        Raises:
            RuntimeError: 合併失敗時
        """
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', audio_codec,
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to merge audio and video: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def create_video_from_image(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 30
    ) -> str:
        """從圖片和音訊建立影片
        
        影片長度將匹配音訊長度。
        
        Args:
            image_path: 輸入圖片路徑
            audio_path: 輸入音訊路徑
            output_path: 輸出影片路徑
            fps: 幀率（預設: 30）
            
        Returns:
            輸出影片路徑
            
        Raises:
            RuntimeError: 建立失敗時
        """
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-tune', 'stillimage',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-r', str(fps),
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to create video from image: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def add_bgm(
        self,
        video_path: str,
        bgm_path: str,
        output_path: str,
        volume: float = 0.2,
        loop: bool = True
    ) -> str:
        """加入背景音樂
        
        Args:
            video_path: 輸入影片路徑
            bgm_path: 背景音樂路徑
            output_path: 輸出影片路徑
            volume: BGM 音量相對於原始音訊（0.0 到 1.0）
            loop: 如果 BGM 比影片短，是否循環播放
            
        Returns:
            輸出影片路徑
        """
        inputs = ['-i', video_path]
        
        bgm_opts = []
        if loop:
            bgm_opts.extend(['-stream_loop', '-1'])
        bgm_opts.extend(['-i', bgm_path])
        
        inputs.extend(bgm_opts)
        
        filter_complex = (
            f"[1:a]volume={volume}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first[a]"
        )
        
        cmd = [
            'ffmpeg',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '0:v',
            '-map', '[a]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to add BGM: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
            
    def concat_audio(
        self,
        audio_paths: List[str],
        output_path: str,
        audio_codec: str = 'libmp3lame'
    ) -> str:
        """合併多個音訊檔案
        
        使用 concat demuxer 並重新編碼以確保有效的輸出流和正確的時長標頭。
        
        Args:
            audio_paths: 要合併的音訊路徑列表
            output_path: 輸出音訊路徑
            audio_codec: 音訊編碼（預設: 'libmp3lame'）
            
        Returns:
            合併後的音訊路徑
            
        Raises:
            RuntimeError: 合併失敗時
        """
        if not audio_paths:
            raise ValueError("audio_paths cannot be empty")
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            filelist_path = f.name
            for audio_path in audio_paths:
                abs_path = os.path.abspath(audio_path)
                f.write(f"file '{abs_path}'\n")
        
        try:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c:a', audio_codec,
                '-y',
                output_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to concatenate audio files: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            # Clean up temp file
            try:
                os.unlink(filelist_path)
            except:
                pass

    def video_to_gif(
        self,
        video_path: str,
        output_path: str,
        fps: int = 12,
        max_colors: int = 256,
        scale_width: int = 512,
        optimize: bool = True
    ) -> str:
        """將影片轉換為優化的 GIF
        
        Args:
            video_path: 輸入影片路徑
            output_path: 輸出 GIF 路徑
            fps: GIF 幀率（預設: 12）
            max_colors: 調色板最大顏色數（預設: 256）
            scale_width: 縮放寬度，高度自動計算（預設: 512）
            optimize: 是否優化 GIF 大小（預設: True）
            
        Returns:
            輸出 GIF 路徑
        """
        try:
            if optimize:
                palette_path = output_path.replace('.gif', '_palette.png')
                palette_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vf', f'fps={fps},scale={scale_width}:-1:flags=lanczos,palettegen=max_colors={max_colors}',
                    '-y',
                    palette_path
                ]
                
                subprocess.run(
                    palette_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                gif_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', palette_path,
                    '-lavfi', f'fps={fps},scale={scale_width}:-1:flags=lanczos[x];[x][1:v]paletteuse',
                    '-y',
                    output_path
                ]
                
                subprocess.run(
                    gif_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                try:
                    os.unlink(palette_path)
                except:
                    pass
            else:
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vf', f'fps={fps},scale={scale_width}:-1:flags=lanczos',
                    '-y',
                    output_path
                ]
                
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to convert video to GIF: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def extract_first_frame(
        self,
        video_path: str,
        output_path: str
    ) -> str:
        """提取影片第一幀
        
        Args:
            video_path: 輸入影片路徑
            output_path: 輸出圖片路徑
            
        Returns:
            提取的圖片路徑
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vframes', '1',
                '-y',
                output_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to extract first frame: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _get_gif_fps(self, gif_path: str, timeout: int = 30) -> Optional[float]:
        """取得 GIF 的 FPS
        
        Args:
            gif_path: GIF 檔案路徑
            timeout: 超時時間（秒）
            
        Returns:
            FPS 值，如果無法確定則返回 None
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                gif_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                timeout=timeout
            )
            
            frame_rate = result.stdout.strip()
            if frame_rate and '/' in frame_rate:
                num, den = map(int, frame_rate.split('/'))
                if den > 0:
                    fps = num / den
                    return fps
            
            return None
        except (subprocess.TimeoutExpired, Exception):
            return None

    def gif_to_mp4(
        self,
        gif_path: str,
        output_path: str,
        fps: Optional[float] = None,
        loop: int = 0,
        pix_fmt: str = 'yuv420p',
        timeout: int = 120
    ) -> str:
        """將 GIF 轉換為 MP4
        
        注意：不使用 -stream_loop，因為會導致轉換卡住。
        對於 Instagram，只需要將 GIF 轉換為 MP4 一次即可。
        
        Args:
            gif_path: 輸入 GIF 路徑
            output_path: 輸出 MP4 路徑
            fps: 幀率（None = 自動偵測，預設: None）
            loop: 已棄用，保留以維持 API 相容性
            pix_fmt: 像素格式（預設: 'yuv420p'）
            timeout: 超時時間（秒，預設: 120）
            
        Returns:
            輸出 MP4 路徑
        """
        try:
            if not os.path.exists(gif_path):
                raise FileNotFoundError(f"GIF 檔案不存在: {gif_path}")
            
            if fps is None:
                detected_fps = self._get_gif_fps(gif_path)
                if detected_fps:
                    fps = detected_fps
                    self.logger.debug(f"偵測到 GIF FPS: {fps}")
                else:
                    fps = 10
                    self.logger.debug(f"使用預設 FPS: {fps}")
            
            cmd = [
                'ffmpeg',
                '-i', gif_path,
                '-vf', f'fps={fps}',
                '-c:v', 'libx264',
                '-pix_fmt', pix_fmt,
                '-movflags', 'faststart',
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=timeout
            )
            
            if not os.path.exists(output_path):
                stderr_output = result.stderr.decode('utf-8', errors='ignore') if isinstance(result.stderr, bytes) else str(result.stderr)
                raise RuntimeError(f"轉換完成但輸出檔案不存在: {output_path}\nFFmpeg stderr: {stderr_output}")
            
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                stderr_output = result.stderr.decode('utf-8', errors='ignore') if isinstance(result.stderr, bytes) else str(result.stderr)
                raise RuntimeError(f"轉換完成但輸出檔案為空: {output_path}\nFFmpeg stderr: {stderr_output}")
            
            self.logger.info(f"GIF 轉換為 MP4 完成: {output_path} (FPS: {fps}, 大小: {output_size / 1024 / 1024:.2f} MB)")
            return output_path
            
        except subprocess.TimeoutExpired as e:
            error_msg = f"GIF 轉 MP4 轉換超時（超過 {timeout} 秒）: {gif_path}"
            self.logger.error(error_msg)
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise RuntimeError(error_msg) from e
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else str(e.stderr)
            error_msg = f"GIF 轉 MP4 轉換失敗: {gif_path}\nFFmpeg stderr: {stderr_output}"
            self.logger.error(error_msg)
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise RuntimeError(error_msg) from e
        except FileNotFoundError:
            raise
        except Exception as e:
            error_msg = f"GIF 轉 MP4 轉換時發生錯誤: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise RuntimeError(error_msg) from e

    def optimize_gif(
        self,
        video_path: str,
        output_path: str,
        fps: int = 12,
        max_colors: int = 256,
        scale_width: int = 512,
        minterpolate: bool = True
    ) -> str:
        """優化 GIF，使用幀插值使動畫更流暢
        
        Args:
            video_path: 輸入影片路徑
            output_path: 輸出 GIF 路徑
            fps: GIF 幀率（預設: 12）
            max_colors: 調色板最大顏色數（預設: 256）
            scale_width: 縮放寬度（預設: 512）
            minterpolate: 是否使用幀插值（預設: True）
            
        Returns:
            輸出 GIF 路徑
        """
        try:
            palette_path = output_path.replace('.gif', '_palette.png')
            
            if minterpolate:
                palette_filter = f'minterpolate=fps={fps * 2}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,fps={fps},scale={scale_width}:-1:flags=lanczos,palettegen=max_colors={max_colors}'
            else:
                palette_filter = f'fps={fps},scale={scale_width}:-1:flags=lanczos,palettegen=max_colors={max_colors}'
            
            palette_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', palette_filter,
                '-y',
                palette_path
            ]
            
            subprocess.run(
                palette_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            if minterpolate:
                gif_filter = f'minterpolate=fps={fps * 2}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,fps={fps},scale={scale_width}:-1:flags=lanczos[x];[x][1:v]paletteuse'
            else:
                gif_filter = f'fps={fps},scale={scale_width}:-1:flags=lanczos[x];[x][1:v]paletteuse'
            
            gif_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', palette_path,
                '-lavfi', gif_filter,
                '-y',
                output_path
            ]
            
            subprocess.run(
                gif_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            try:
                os.unlink(palette_path)
            except:
                pass
            
            self.logger.info(f"已優化 GIF: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to optimize GIF: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e