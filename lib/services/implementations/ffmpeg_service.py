"""FFmpeg Service - Video processing operations for long-form video generation"""
import subprocess
import os
import tempfile
from typing import List, Optional, Literal
from pathlib import Path
import logging


class FFmpegService:
    """
    FFmpeg service for video processing operations.
    
    Provides core video manipulation functions needed for long-form video generation:
    - Extract last frame from video segments
    - Concatenate multiple video segments
    - Merge audio with video
    - Create video from static image + audio
    
    All operations use ffmpeg command-line tool.
    """
    
    def __init__(self):
        """Initialize FFmpeg service and verify ffmpeg is available"""
        self.logger = logging.getLogger(__name__)
        self._check_ffmpeg_installed()
    
    def _check_ffmpeg_installed(self):
        """Verify ffmpeg is installed and accessible"""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            self.logger.info("FFmpeg is available")
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
        """
        Extract the last frame from a video file.
        
        This is critical for long-form video generation as each segment
        uses the previous segment's last frame as conditioning.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output image file (e.g., 'frame_last.png')
        
        Returns:
            Path to extracted frame image
            
        Raises:
            RuntimeError: If frame extraction fails
            
        Example:
            >>> ffmpeg_service = FFmpegService()
            >>> last_frame = ffmpeg_service.extract_last_frame(
            ...     'segment_1.mp4',
            ...     'segment_1_last_frame.png'
            ... )
        """
        try:
            # Get total frame count
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
            
            # Extract last frame
            extract_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'select=eq(n\\,{last_frame_idx})',
                '-vframes', '1',
                '-y',  # Overwrite output file
                output_path
            ]
            
            subprocess.run(
                extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            self.logger.info(f"Extracted last frame from {video_path} to {output_path}")
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
        """
        Concatenate multiple video files into one.
        
        Args:
            video_paths: List of video file paths to concatenate (in order)
            output_path: Path for output concatenated video
            method: Concatenation method
                - 'demuxer': Fast, no re-encoding (requires same codec)
                - 'filter': Slower but handles different formats
        
        Returns:
            Path to concatenated video
            
        Raises:
            RuntimeError: If concatenation fails
            
        Example:
            >>> segments = ['seg1.mp4', 'seg2.mp4', 'seg3.mp4']
            >>> final = ffmpeg_service.concat_videos(segments, 'final.mp4')
        """
        if not video_paths:
            raise ValueError("video_paths cannot be empty")
        
        if len(video_paths) == 1:
            # Only one video, just copy it
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
        """
        Concatenate using concat demuxer (fast, no re-encoding).
        
        FFmpeg equivalent:
            ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
        """
        # Create temporary file list
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            filelist_path = f.name
            for video_path in video_paths:
                # Convert to absolute path and escape special characters
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
            
            self.logger.info(f"Concatenated {len(video_paths)} videos to {output_path}")
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
        """
        Concatenate using concat filter (slower but handles different formats).
        
        FFmpeg equivalent:
            ffmpeg -i v1.mp4 -i v2.mp4 
                   -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
                   -map "[v]" -map "[a]" output.mp4
        """
        # Build filter complex
        inputs = []
        for path in video_paths:
            inputs.extend(['-i', path])
        
        # Build concat filter
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
            
            self.logger.info(f"Concatenated {len(video_paths)} videos using filter to {output_path}")
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
        """
        Merge audio file with video file.
        
        If video already has audio, it will be replaced with the new audio.
        
        Args:
            video_path: Path to input video file
            audio_path: Path to input audio file
            output_path: Path for output video with merged audio
            audio_codec: Audio codec to use (default: 'aac')
        
        Returns:
            Path to output video
            
        Raises:
            RuntimeError: If merging fails
            
        Example:
            >>> result = ffmpeg_service.merge_audio_video(
            ...     'video.mp4',
            ...     'narration.mp3',
            ...     'video_with_audio.mp4'
            ... )
        """
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', audio_codec,  # Encode audio
            '-map', '0:v:0',  # Map video from first input
            '-map', '1:a:0',  # Map audio from second input
            '-shortest',  # Finish when shortest stream ends
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
            
            self.logger.info(f"Merged audio from {audio_path} with video {video_path}")
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
        """
        Create video from static image and audio.
        
        Video duration will match audio duration.
        Useful for creating video segments from still images with narration.
        
        Args:
            image_path: Path to input image
            audio_path: Path to input audio file
            output_path: Path for output video
            fps: Frames per second (default: 30)
        
        Returns:
            Path to output video
            
        Raises:
            RuntimeError: If video creation fails
            
        Example:
            >>> video = ffmpeg_service.create_video_from_image(
            ...     'image.png',
            ...     'narration.mp3',
            ...     'segment.mp4'
            ... )
        """
        cmd = [
            'ffmpeg',
            '-loop', '1',  # Loop the image
            '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-tune', 'stillimage',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',  # Duration = audio duration
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
            
            self.logger.info(f"Created video from image {image_path} with audio {audio_path}")
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
        """
        Add background music to video.
        
        Args:
            video_path: Path to input video
            bgm_path: Path to BGM audio file
            output_path: Path for output video
            volume: BGM volume relative to original audio (0.0 to 1.0)
            loop: Whether to loop BGM if shorter than video
            
        Returns:
            Path to output video
        """
        # Build inputs
        inputs = ['-i', video_path]
        
        # BGM input options
        bgm_opts = []
        if loop:
            bgm_opts.extend(['-stream_loop', '-1'])
        bgm_opts.extend(['-i', bgm_path])
        
        inputs.extend(bgm_opts)
        
        # Filter complex for mixing
        # [1:a]volume=0.2[bgm];[0:a][bgm]amix=inputs=2:duration=first[a]
        filter_complex = (
            f"[1:a]volume={volume}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first[a]"
        )
        
        cmd = [
            'ffmpeg',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '0:v',   # Keep original video
            '-map', '[a]',   # Use mixed audio
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # Encode audio
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
            
            self.logger.info(f"Added BGM {bgm_path} to {video_path}")
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
        """
        Concatenate multiple audio files into one.
        
        Uses concat demuxer with re-encoding to ensure valid output stream
        and correct duration headers (fixes 'Estimating duration from bitrate' error).
        
        Args:
            audio_paths: List of audio file paths to concatenate
            output_path: Path for output concatenated audio
            audio_codec: Audio codec to use (default: 'libmp3lame')
            
        Returns:
            Path to concatenated audio
            
        Raises:
            RuntimeError: If concatenation fails
        """
        if not audio_paths:
            raise ValueError("audio_paths cannot be empty")
            
        # Create temporary file list
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            delete=False,
            encoding='utf-8'
        ) as f:
            filelist_path = f.name
            for audio_path in audio_paths:
                # Convert to absolute path and escape special characters
                abs_path = os.path.abspath(audio_path)
                f.write(f"file '{abs_path}'\n")
        
        try:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c:a', audio_codec,  # Re-encode audio
                '-y',
                output_path
            ]
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            self.logger.info(f"Concatenated {len(audio_paths)} audio files to {output_path}")
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
        """
        Convert video to optimized GIF.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output GIF file
            fps: Frames per second for GIF (default: 12)
            max_colors: Maximum colors in palette (default: 256)
            scale_width: Scale width, height auto-calculated (default: 512)
            optimize: Whether to optimize GIF size (default: True)
        
        Returns:
            Path to output GIF
        """
        try:
            if optimize:
                palette_path = output_path.replace('.gif', '_palette.png')
                
                # Pass 1: Generate palette
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
                
                # Pass 2: Generate GIF with palette
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
            
            self.logger.info(f"Converted {video_path} to GIF: {output_path}")
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
        """
        Extract the first frame from a video file.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output image file
        
        Returns:
            Path to extracted frame image
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
            
            self.logger.info(f"Extracted first frame from {video_path} to {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to extract first frame: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _get_gif_fps(self, gif_path: str) -> float:
        """
        Get FPS from GIF file using ffprobe.
        
        Args:
            gif_path: Path to GIF file
        
        Returns:
            FPS value, or None if cannot be determined
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
                text=True
            )
            
            frame_rate = result.stdout.strip()
            if frame_rate and '/' in frame_rate:
                num, den = map(int, frame_rate.split('/'))
                if den > 0:
                    fps = num / den
                    return fps
            
            return None
        except Exception as e:
            self.logger.debug(f"Could not get FPS from GIF {gif_path}: {e}")
            return None

    def gif_to_mp4(
        self,
        gif_path: str,
        output_path: str,
        fps: Optional[float] = None,
        loop: int = 0,
        pix_fmt: str = 'yuv420p'
    ) -> str:
        """
        Convert GIF to MP4 video.
        
        Args:
            gif_path: Path to input GIF file
            output_path: Path for output MP4 file
            fps: Frames per second (None = auto-detect from GIF, default: None)
            loop: Number of loops (0 = infinite, -1 = no loop, default: 0)
            pix_fmt: Pixel format (default: 'yuv420p' for compatibility)
        
        Returns:
            Path to output MP4
        """
        try:
            # Auto-detect FPS from GIF if not provided
            if fps is None:
                detected_fps = self._get_gif_fps(gif_path)
                if detected_fps:
                    fps = detected_fps
                    self.logger.debug(f"Detected GIF FPS: {fps}")
                else:
                    fps = 10  # Default FPS for GIFs
                    self.logger.debug(f"Using default FPS: {fps}")
            
            cmd = ['ffmpeg']
            
            # -stream_loop must be before -i
            if loop != -1:
                if loop == 0:
                    cmd.extend(['-stream_loop', '-1'])
                else:
                    cmd.extend(['-stream_loop', str(loop)])
            
            cmd.extend([
                '-i', gif_path,
                '-vf', f'fps={fps}',
                '-c:v', 'libx264',
                '-pix_fmt', pix_fmt,
                '-y',
                output_path
            ])
            
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            self.logger.info(f"Converted {gif_path} to MP4: {output_path} (FPS: {fps})")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to convert GIF to MP4: {e.stderr}"
            self.logger.error(error_msg)
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
        """
        Optimize GIF with frame interpolation for smoother animation.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output GIF file
            fps: Frames per second for GIF (default: 12)
            max_colors: Maximum colors in palette (default: 256)
            scale_width: Scale width, height auto-calculated (default: 512)
            minterpolate: Whether to use frame interpolation for smoother motion (default: True)
        
        Returns:
            Path to output GIF
        """
        try:
            palette_path = output_path.replace('.gif', '_palette.png')
            
            # Pass 1: Generate palette
            if minterpolate:
                # Use frame interpolation for smoother motion
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
            
            # Pass 2: Generate optimized GIF with palette
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
            
            self.logger.info(f"Optimized GIF created: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to optimize GIF: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e