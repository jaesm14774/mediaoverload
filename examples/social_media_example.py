import os
import sys
from typing import Dict, Any

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.social_media import (
    SocialMediaManager, 
    MediaPost, 
    InstagramPlatform, 
    TwitterPlatform
)

def main():
    print("🚀 Social Media Manager Example")
    print("-" * 30)

    # 1. Initialize the Manager
    manager = SocialMediaManager()
    print("✅ Manager initialized")

    # 2. Configure and Register Platforms
    # Note: In a real scenario, you would provide actual config paths
    # containing credentials.json and other settings.
    
    # Example configuration paths (these are placeholders)
    instagram_config_path = "configs/instagram"
    twitter_config_path = "configs/twitter"
    
    # We'll wrap this in a try-except block because we don't have real credentials here
    try:
        # Initialize platforms
        # The prefix is used to find specific config files (e.g., 'my_account_credentials.json')
        instagram = InstagramPlatform(instagram_config_path, prefix="my_account")
        twitter = TwitterPlatform(twitter_config_path, prefix="my_account")
        
        # Register them to the manager
        # This will attempt to authenticate
        manager.register_platform("instagram", instagram)
        manager.register_platform("twitter", twitter)
        print("✅ Platforms registered and authenticated")
        
    except Exception as e:
        print(f"⚠️  Note: Platform registration failed (expected without real credentials): {e}")
        print("   Continuing with mock demonstration...")
        
        # For demonstration, we can mock the platforms if needed, 
        # or just show how the code would look.

    # 3. Create a Post
    # Create a MediaPost object with your content
    post = MediaPost(
        media_paths=["output_media/example_image.png"],
        caption="Check out this amazing generated art! 🎨 #aiart #creative",
        hashtags="#ai #generative",
        additional_params={
            "location": "Digital World"
        }
    )
    print(f"\n📦 Created post object:")
    print(f"   Caption: {post.caption}")
    print(f"   Media: {post.media_paths}")

    # 4. Upload to Platforms
    print("\n📤 Uploading to platforms...")
    
    # Option A: Upload to all registered platforms
    # results = manager.upload_to_all(post)
    
    # Option B: Upload to specific platform
    # result_ig = manager.upload_to_platform("instagram", post)
    
    print("   (Upload calls are commented out to prevent errors without credentials)")
    print("\n✨ Example completed!")

if __name__ == "__main__":
    main()
