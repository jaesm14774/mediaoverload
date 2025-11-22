from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class MediaPost:
    """Base structure for social media posts"""
    media_paths: List[str]
    caption: str
    hashtags: Optional[str] = None
    additional_params: Dict[str, Any] = None
