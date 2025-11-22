from abc import ABC, abstractmethod
from .models import MediaPost

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
