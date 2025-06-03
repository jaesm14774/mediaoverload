from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
import time
import random
from instagrapi.types import StoryLink

@dataclass
class MediaPost:
    """Base structure for social media posts"""
    images: List[str]
    caption: str
    hashtags: Optional[str] = None
    additional_params: Dict[str, Any] = None

class SocialMediaPlatform(ABC):
    """Abstract base class for social media platforms"""
    
    @abstractmethod
    def authenticate(self) -> None:
        """Authenticate with the platform"""
        pass
    
    @abstractmethod
    def upload_post(self, post: MediaPost) -> bool:
        """Upload content to the platform"""
        pass
    
    @abstractmethod
    def load_config(self) -> None:
        """Load platform-specific configurations"""
        pass


class InstagramPlatform(SocialMediaPlatform):
    """Instagram-specific implementation"""
    def __init__(self, config_folder_path: str, prefix=''):
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.load_config()
        self.authenticate()
        
    def authenticate(self) -> None:
        from instagrapi import Client
        self.client = Client()
        self.settings_path = f"{self.config_path}/{os.getenv('IG_ACCOUNT_COOKIE_FILE_PATH')}"
        if not os.path.exists(self.settings_path):
            self._create_new_session()
        else:
            self.client.load_settings(self.settings_path)
    
    def _create_new_session(self) -> None:
        self.client.login(
            username=os.getenv('IG_USERNAME'),
            password=os.getenv('IG_PASSWORD')
        )
        self.client.dump_settings(self.settings_path)
    
    def upload_post(self, post: MediaPost) -> bool:
        try:
            caption = f"{post.caption}\n{post.hashtags}" if post.hashtags else post.caption
            
            if len(post.images) == 1:
                media = self.client.photo_upload(
                    path=post.images[0],
                    caption=caption
                )
            else:
                media = self.client.album_upload(
                    paths=post.images,
                    caption=caption
                )
            
            # Handle Instagram-specific features
            if post.additional_params.get('share_to_story'):
                time.sleep(5)
                self._share_to_story(post.images, media.code)
            
            return True
        except Exception as e:
            print(f"Instagram upload failed: {str(e)}")
            return False

    def _share_to_story(self, images: List[str], post_code: str):
        """分享貼文到限時動態"""
        try:
            self.client.photo_upload_to_story(
                random.choice(images),
                caption="查看最新貼文 ⬆️",
                links=[StoryLink(webUri=f'https://www.instagram.com/p/{post_code}')]
            )
        except Exception as e:
            print(f"分享限時動態失敗: {str(e)}")

    
    def load_config(self) -> None:
        self.config_path = f"{self.config_folder_path}/{self.prefix}" if self.prefix else f"{self.config_folder_path}"
        load_dotenv(f'{self.config_path}/ig.env')


class TwitterPlatform(SocialMediaPlatform):
    """Twitter-specific implementation"""
    
    def authenticate(self) -> None:
        # Implementation for Twitter authentication
        pass
    
    def upload_post(self, post: MediaPost) -> bool:
        # Implementation for Twitter upload
        pass
    
    def load_config(self) -> None:
        pass


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
