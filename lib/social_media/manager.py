from typing import Dict, Optional, List

from .models import MediaPost
from .base import SocialMediaPlatform

class SocialMediaManager:
    """Manages social media platform interactions"""
    def __init__(self):
        self.platforms: Dict[str, SocialMediaPlatform] = {}
    
    def register_platform(self, name: str, platform: SocialMediaPlatform) -> None:
        """Register a new social media platform"""
        self.platforms[name] = platform
        platform.authenticate()
    
    def upload_to_platform(self, platform_name: str, post: MediaPost) -> bool:
        """Upload content to a specific platform"""
        if platform_name not in self.platforms:
            raise ValueError(f"Platform {platform_name} not registered")
        
        return self.platforms[platform_name].upload_post(post)
    
    def upload_to_all(self, post: MediaPost) -> Dict[str, bool]:
        """Upload content to all registered platforms"""
        results = {}
        for platform_name, platform in self.platforms.items():
            results[platform_name] = platform.upload_post(post)
        return results

class SocialMediaMixin:
    """Mixin for adding social media capabilities to Process classes"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.social_media_manager = SocialMediaManager()
        
    def register_social_media(self, platforms: Dict[str, tuple]) -> None:
        """Register social media platforms with their configs"""
        for platform_name, (platform_class, config_folder_path, prefix) in platforms.items():
            platform = platform_class(config_folder_path, prefix)
            self.social_media_manager.register_platform(platform_name, platform)
    
    def upload_to_social_media(self, post: MediaPost, platforms: Optional[List[str]] = None) -> Dict[str, bool]:
        """Upload content to specified or all platforms"""
        if platforms:
            results = {}
            for platform in platforms:
                results[platform] = self.social_media_manager.upload_to_platform(platform, post)
            return results
        return self.social_media_manager.upload_to_all(post)
