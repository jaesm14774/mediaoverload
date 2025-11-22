import os
import time
import tweepy
from dotenv import load_dotenv

from .models import MediaPost
from .base import SocialMediaPlatform

class TwitterPlatform(SocialMediaPlatform):
    """Twitter-specific implementation"""
    
    def __init__(self, config_folder_path: str, prefix=''):
        from utils.logger import setup_logger
        self.logger = setup_logger(__name__)
        self.config_folder_path = config_folder_path
        self.prefix = prefix
        self.client = None  # v1.1 API client
        self.client_v2 = None  # v2 API client
        self.client_v1 = None  # v1.1 API client (for media upload)
        self.load_config()
        self.authenticate()
    
    def authenticate(self) -> None:
        """使用 tweepy 進行 Twitter 認證（支援 API v2 和 v1.1）"""
        try:
            # 檢查必要的環境變數
            api_key = os.getenv('TWITTER_API_KEY')
            api_secret = os.getenv('TWITTER_API_SECRET')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            
            if not all([api_key, api_secret, access_token, access_token_secret]):
                missing = [k for k, v in {
                    'TWITTER_API_KEY': api_key,
                    'TWITTER_API_SECRET': api_secret,
                    'TWITTER_ACCESS_TOKEN': access_token,
                    'TWITTER_ACCESS_TOKEN_SECRET': access_token_secret
                }.items() if not v]
                error_msg = f"缺少必要的 Twitter API 憑證: {', '.join(missing)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 創建 OAuth handler（v1.1 和 v2 都需要）
            auth = tweepy.OAuth1UserHandler(
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            
            # 創建 v1.1 API 客戶端（用於媒體上傳，免費方案可用）
            self.client_v1 = tweepy.API(auth, wait_on_rate_limit=False)
            
            # 使用 Twitter API v2 Client（推薦，支援免費方案）
            # v2 API 使用 OAuth 1.0a User Context
            try:
                self.client_v2 = tweepy.Client(
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    bearer_token=bearer_token if bearer_token else None,
                    wait_on_rate_limit=False
                )
                self.logger.debug("Twitter API v2 客戶端已初始化（不進行連線測試以節省 API 呼叫次數）")
                self.logger.debug("v1.1 API 客戶端已初始化（用於媒體上傳）")
                self.client = None  # 標記使用 v2
            except Exception as e:
                self.logger.warning(f"無法初始化 Twitter API v2，將使用 v1.1: {str(e)}")
                # 備用方案：使用 v1.1 API
                self.client = self.client_v1  # 標記使用 v1.1
                self.logger.debug("Twitter API v1.1 客戶端已初始化（不進行連線測試以節省 API 呼叫次數）")
        except ImportError:
            error_msg = "請安裝 tweepy 套件: pip install tweepy"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            self.logger.error(f"Twitter 認證錯誤: {str(e)}", exc_info=True)
            raise
    
    def upload_post(self, post: MediaPost) -> bool:
        """上傳內容到 Twitter（優先使用 API v2，備用 v1.1）"""
        try:
            if not self.client_v2 and not self.client:
                error_msg = "Twitter 客戶端未初始化，請先進行認證"
                self.logger.error(error_msg)
                return False
            
            # 組合推文內容
            text = post.caption or ''
            if post.hashtags:
                text = f"{text}\n{post.hashtags}" if text else post.hashtags
            
            # Twitter 推文長度限制為 280 字元
            original_length = len(text)
            if original_length > 280:
                text = text[:277] + "..."
                self.logger.warning(f"推文內容過長 ({original_length} 字元)，已截斷為 280 字元")
            
            self.logger.info(f"準備發布推文，內容長度: {len(text)} 字元，媒體數量: {len(post.media_paths) if post.media_paths else 0}")
            
            # 優先使用 API v2
            if self.client_v2:
                return self._upload_with_v2(post, text)
            # 備用 v1.1 API
            elif self.client:
                return self._upload_with_v1(post, text)
            else:
                error_msg = "沒有可用的 Twitter API 客戶端"
                self.logger.error(error_msg)
                return False
        except tweepy.TweepyException as e:
            error_msg = f"Twitter API 錯誤: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False
        except Exception as e:
            error_msg = f"Twitter 上傳失敗: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return False
    
    def _upload_with_v2(self, post: MediaPost, text: str) -> bool:
        """使用 Twitter API v2 發布推文"""
        # 處理媒體上傳（v2 需要使用 v1.1 的媒體上傳端點）
        media_ids = []
        if post.media_paths:
            # 需要 v1.1 API 來上傳媒體
            if not self.client_v1:
                self.logger.warning("需要 v1.1 API 來上傳媒體，但 v1.1 客戶端未初始化")
                # 嘗試初始化 v1.1 客戶端
                try:
                    api_key = os.getenv('TWITTER_API_KEY')
                    api_secret = os.getenv('TWITTER_API_SECRET')
                    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
                    access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
                    
                    auth = tweepy.OAuth1UserHandler(
                        consumer_key=api_key,
                        consumer_secret=api_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret
                    )
                    self.client_v1 = tweepy.API(auth, wait_on_rate_limit=False)  # 已啟用速率限制處理
                except Exception as e:
                    self.logger.error(f"無法初始化 v1.1 API 客戶端: {str(e)}")
            
            if self.client_v1:
                # 上傳所有媒體
                for idx, media_path in enumerate(post.media_paths, 1):
                    if not os.path.exists(media_path):
                        self.logger.warning(f"媒體檔案不存在: {media_path}")
                        continue
                    
                    try:
                        self.logger.info(f"正在上傳媒體 {idx}/{len(post.media_paths)}: {media_path}")
                        
                        # 在媒體上傳之間添加間隔，避免觸發速率限制
                        if idx > 1:
                            time.sleep(2)  # 每次上傳間隔 2 秒
                        
                        # 判斷媒體類型
                        if media_path.endswith('.mp4'):
                            # 影片上傳
                            media = self.client_v1.media_upload(
                                media_path,
                                media_category='tweet_video'
                            )
                            self.logger.info(f"影片上傳成功，media_id: {media.media_id}，等待處理...")
                            self._wait_for_video_processing(media.media_id, use_v1=True)
                            self.logger.info(f"影片處理完成")
                        else:
                            # 圖片上傳
                            media = self.client_v1.media_upload(media_path)
                            self.logger.info(f"圖片上傳成功，media_id: {media.media_id}")
                        
                        media_ids.append(media.media_id)
                    except tweepy.TooManyRequests as e:
                        wait_time = self._extract_rate_limit_wait_time(e)
                        self.logger.warning(f"媒體上傳遇到速率限制，等待 {wait_time} 秒後重試")
                        time.sleep(wait_time)
                        # 重試上傳
                        try:
                            if media_path.endswith('.mp4'):
                                media = self.client_v1.media_upload(
                                    media_path,
                                    media_category='tweet_video'
                                )
                                self._wait_for_video_processing(media.media_id, use_v1=True)
                            else:
                                media = self.client_v1.media_upload(media_path)
                            media_ids.append(media.media_id)
                            self.logger.info(f"媒體上傳重試成功，media_id: {media.media_id}")
                        except Exception as retry_error:
                            error_msg = f"媒體上傳重試失敗 {media_path}: {str(retry_error)}"
                            self.logger.error(error_msg, exc_info=True)
                            continue
                    except Exception as e:
                        error_msg = f"上傳媒體失敗 {media_path}: {str(e)}"
                        self.logger.error(error_msg, exc_info=True)
                        continue
                
                # Twitter 最多支援 4 張圖片
                if len(media_ids) > 4:
                    self.logger.warning(f"媒體數量超過限制 (4)，將只使用前 4 個")
                    media_ids = media_ids[:4]
        
        # 使用 v2 API 發布推文（帶重試機制）
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if media_ids:
                    self.logger.info(f"使用 API v2 發布推文，包含 {len(media_ids)} 個媒體")
                    tweet = self.client_v2.create_tweet(text=text, media_ids=media_ids)
                else:
                    self.logger.info("使用 API v2 發布純文字推文")
                    tweet = self.client_v2.create_tweet(text=text)
                
                if tweet.data:
                    tweet_id = tweet.data['id']
                    # 避免不必要的 get_me() 調用，直接使用 tweet_id
                    # 如果需要用戶名，可以從 tweet 對象獲取或緩存
                    self.logger.info(f"推文發布成功，Tweet ID: {tweet_id}, URL: https://twitter.com/i/web/status/{tweet_id}")
                    return True
                else:
                    self.logger.error("推文發布失敗：未返回推文數據")
                    return False
            except tweepy.TooManyRequests as e:
                retry_count += 1
                wait_time = self._extract_rate_limit_wait_time(e)
                
                if retry_count < max_retries:
                    self.logger.warning(f"遇到速率限制 (429)，等待 {wait_time} 秒後重試 ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Twitter API v2 發布失敗：已達到最大重試次數，速率限制未解除"
                    self.logger.error(error_msg)
                    raise
            except tweepy.TweepyException as e:
                error_msg = f"Twitter API v2 發布失敗: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                
                # 檢查是否為權限問題
                if isinstance(e, tweepy.Forbidden):
                    self.logger.warning("API v2 權限不足，可能原因：")
                    self.logger.warning("1. 應用程式沒有 'Read and Write' 權限")
                    self.logger.warning("2. 請在 Twitter Developer Portal 檢查應用程式權限設定")
                    self.logger.warning("3. 需要重新生成 Access Token 和 Access Token Secret")
                
                # 如果 v2 失敗，嘗試使用 v1.1（如果有）
                # 注意：媒體已經上傳，只需要發布推文
                if self.client_v1:
                    self.logger.debug("嘗試使用 API v1.1 作為備用方案發布推文")
                    try:
                        # 使用 v1.1 API 發布推文（媒體已上傳）
                        if media_ids:
                            self.logger.info(f"使用 API v1.1 發布推文，包含 {len(media_ids)} 個媒體")
                            tweet = self.client_v1.update_status(status=text, media_ids=media_ids)
                        else:
                            self.logger.info("使用 API v1.1 發布純文字推文")
                            tweet = self.client_v1.update_status(status=text)
                        
                        self.logger.info(f"推文發布成功（v1.1），Tweet ID: {tweet.id}, URL: https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}")
                        return True
                    except tweepy.TooManyRequests as v1_error:
                        wait_time = self._extract_rate_limit_wait_time(v1_error)
                        self.logger.warning(f"API v1.1 也遇到速率限制，等待 {wait_time} 秒後重試")
                        time.sleep(wait_time)
                        # 重試一次 v1.1
                        try:
                            if media_ids:
                                tweet = self.client_v1.update_status(status=text, media_ids=media_ids)
                            else:
                                tweet = self.client_v1.update_status(status=text)
                            self.logger.info(f"推文發布成功（v1.1 重試），Tweet ID: {tweet.id}")
                            return True
                        except Exception as retry_error:
                            self.logger.error(f"API v1.1 重試也失敗: {str(retry_error)}")
                            return False
                    except tweepy.TweepyException as v1_error:
                        self.logger.error(f"API v1.1 發布也失敗: {str(v1_error)}", exc_info=True)
                        if isinstance(v1_error, tweepy.Forbidden):
                            self.logger.error("API v1.1 也需要付費方案才能發布推文")
                        return False
                elif self.client:
                    # 如果 self.client 存在（舊的 v1.1 客戶端）
                    self.logger.debug("嘗試使用 API v1.1 作為備用方案")
                    return self._upload_with_v1(post, text)
                else:
                    self.logger.error("沒有可用的備用 API 客戶端")
                    return False
    
    def _upload_with_v1(self, post: MediaPost, text: str) -> bool:
        """使用 Twitter API v1.1 發布推文（需要付費方案）"""
        media_ids = []
        
        if post.media_paths:
            # 上傳所有媒體
            for idx, media_path in enumerate(post.media_paths, 1):
                if not os.path.exists(media_path):
                    self.logger.warning(f"媒體檔案不存在: {media_path}")
                    continue
                
                try:
                    self.logger.info(f"正在上傳媒體 {idx}/{len(post.media_paths)}: {media_path}")
                    
                    # 在媒體上傳之間添加間隔，避免觸發速率限制
                    if idx > 1:
                        time.sleep(2)  # 每次上傳間隔 2 秒
                    
                    if media_path.endswith('.mp4'):
                        media = self.client.media_upload(
                            media_path,
                            media_category='tweet_video'
                        )
                        self.logger.info(f"影片上傳成功，media_id: {media.media_id}，等待處理...")
                        self._wait_for_video_processing(media.media_id, use_v1=True)
                        self.logger.info(f"影片處理完成")
                    else:
                        media = self.client.media_upload(media_path)
                        self.logger.info(f"圖片上傳成功，media_id: {media.media_id}")
                    
                    media_ids.append(media.media_id)
                except tweepy.TooManyRequests as e:
                    wait_time = self._extract_rate_limit_wait_time(e)
                    self.logger.warning(f"媒體上傳遇到速率限制，等待 {wait_time} 秒後重試")
                    time.sleep(wait_time)
                    # 重試上傳
                    try:
                        if media_path.endswith('.mp4'):
                            media = self.client.media_upload(
                                media_path,
                                media_category='tweet_video'
                            )
                            self._wait_for_video_processing(media.media_id, use_v1=True)
                        else:
                            media = self.client.media_upload(media_path)
                        media_ids.append(media.media_id)
                        self.logger.info(f"媒體上傳重試成功，media_id: {media.media_id}")
                    except Exception as retry_error:
                        error_msg = f"媒體上傳重試失敗 {media_path}: {str(retry_error)}"
                        self.logger.error(error_msg, exc_info=True)
                        continue
                except Exception as e:
                    error_msg = f"上傳媒體失敗 {media_path}: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    continue
            
            if len(media_ids) > 4:
                self.logger.warning(f"媒體數量超過限制 (4)，將只使用前 4 個")
                media_ids = media_ids[:4]
        
        # 發布推文（帶重試機制）
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if media_ids:
                    self.logger.info(f"使用 API v1.1 發布推文，包含 {len(media_ids)} 個媒體")
                    tweet = self.client.update_status(status=text, media_ids=media_ids)
                else:
                    self.logger.info("使用 API v1.1 發布純文字推文")
                    tweet = self.client.update_status(status=text)
                
                self.logger.info(f"推文發布成功，Tweet ID: {tweet.id}, URL: https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}")
                return True
            except tweepy.TooManyRequests as e:
                retry_count += 1
                wait_time = self._extract_rate_limit_wait_time(e)
                
                if retry_count < max_retries:
                    self.logger.warning(f"遇到速率限制 (429)，等待 {wait_time} 秒後重試 ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Twitter API v1.1 發布失敗：已達到最大重試次數，速率限制未解除"
                    self.logger.error(error_msg)
                    return False
            except tweepy.TweepyException as e:
                error_msg = f"Twitter API v1.1 發布失敗: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return False
    
    def _extract_rate_limit_wait_time(self, error: tweepy.TooManyRequests) -> int:
        """從 429 錯誤中提取需要等待的時間（秒）"""
        try:
            # 嘗試從響應 headers 中獲取速率限制重置時間
            if hasattr(error, 'response') and error.response is not None:
                headers = error.response.headers
                # Twitter API 通常會在 x-rate-limit-reset header 中提供重置時間（Unix timestamp）
                if 'x-rate-limit-reset' in headers:
                    reset_time = int(headers['x-rate-limit-reset'])
                    current_time = int(time.time())
                    wait_time = max(reset_time - current_time, 0)
                    # 添加一些緩衝時間（15分鐘 + 10秒緩衝）
                    if wait_time > 0:
                        return wait_time + 10
                
                # 如果沒有 reset header，檢查 Retry-After header
                if 'retry-after' in headers:
                    retry_after = int(headers['retry-after'])
                    return retry_after + 10
        except Exception as e:
            self.logger.warning(f"無法從錯誤中提取等待時間: {str(e)}")
        
        # 默認等待時間：15分鐘（900秒）+ 緩衝
        # Twitter API v2 免費方案的速率限制通常是每15分鐘一定數量的請求
        self.logger.info("使用默認等待時間：15分鐘")
        return 900 + 10
    
    def _wait_for_video_processing(self, media_id: str, max_wait_time: int = 300, use_v1: bool = True) -> None:
        """等待影片處理完成"""
        start_time = time.time()
        
        self.logger.info(f"等待影片處理，media_id: {media_id}")
        
        # 使用 v1.1 API 檢查處理狀態
        client = self.client_v1 if use_v1 and self.client_v1 else self.client
        if not client:
            raise Exception("無法檢查影片處理狀態：沒有可用的 API 客戶端")
        
        while time.time() - start_time < max_wait_time:
            try:
                status = client.get_media_upload_status(media_id)
                if status.processing_info:
                    state = status.processing_info.get('state', '')
                    progress = status.processing_info.get('progress_percent', 0)
                    self.logger.debug(f"影片處理狀態: {state}, 進度: {progress}%")
                    
                    if state == 'succeeded':
                        self.logger.info("影片處理成功")
                        return
                    elif state == 'failed':
                        error_msg = status.processing_info.get('error', {}).get('message', '未知錯誤')
                        raise Exception(f"影片處理失敗: {error_msg}")
                    # 等待處理
                    check_after = status.processing_info.get('check_after_secs', 5)
                    time.sleep(check_after)
                else:
                    # 沒有處理資訊，假設已完成
                    self.logger.info("影片處理完成（無處理資訊）")
                    return
            except tweepy.TooManyRequests as e:
                wait_time = self._extract_rate_limit_wait_time(e)
                self.logger.warning(f"檢查影片狀態時遇到速率限制，等待 {wait_time} 秒")
                time.sleep(wait_time)
                # 繼續循環檢查
                continue
            except tweepy.TweepyException as e:
                self.logger.error(f"檢查影片處理狀態時發生錯誤: {str(e)}")
                raise
        
        raise Exception(f"影片處理超時（超過 {max_wait_time} 秒）")
    
    def load_config(self) -> None:
        """載入 Twitter 配置"""
        self.config_path = f"{self.config_folder_path}/{self.prefix}" if self.prefix else f"{self.config_folder_path}"
        load_dotenv(f'{self.config_path}/twitter.env')
