import logging
from urllib.parse import urlparse

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from lib.instagram.mixins.account import AccountMixin
from lib.instagram.mixins.album import DownloadAlbumMixin, UploadAlbumMixin
from lib.instagram.mixins.auth import LoginMixin
from lib.instagram.mixins.challenge import ChallengeResolveMixin
from lib.instagram.mixins.clip import DownloadClipMixin, UploadClipMixin
from lib.instagram.mixins.media import MediaMixin
from lib.instagram.mixins.photo import DownloadPhotoMixin, UploadPhotoMixin
from lib.instagram.mixins.private import PrivateRequestMixin
from lib.instagram.mixins.public import (
    ProfilePublicMixin,
    PublicRequestMixin,
    TopSearchesPublicMixin,
)
from lib.instagram.mixins.story import StoryMixin
from lib.instagram.mixins.totp import TOTPMixin
from lib.instagram.mixins.video import DownloadVideoMixin, UploadVideoMixin

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Used as fallback logger if another is not provided.
DEFAULT_LOGGER = logging.getLogger("instagrapi")


class Client(
    PublicRequestMixin,
    ChallengeResolveMixin,
    PrivateRequestMixin,
    TopSearchesPublicMixin,
    ProfilePublicMixin,
    LoginMixin,
    DownloadPhotoMixin,
    UploadPhotoMixin,
    DownloadVideoMixin,
    UploadVideoMixin,
    DownloadAlbumMixin,
    UploadAlbumMixin,
    MediaMixin,
    AccountMixin,
    StoryMixin,
    DownloadClipMixin,
    UploadClipMixin,
    TOTPMixin,
):
    proxy = None

    def __init__(
        self,
        settings: dict = {},
        proxy: str | None = None,
        delay_range: list | None = None,
        logger=DEFAULT_LOGGER,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.settings = settings
        self.logger = logger
        self.delay_range = delay_range

        self.set_proxy(proxy)

        self.init()

    def set_proxy(self, dsn: str | None):
        if dsn:
            assert isinstance(
                dsn, str
            ), f'Proxy must been string (URL), but now "{dsn}" ({type(dsn)})'
            self.proxy = dsn
            proxy_href = "{scheme}{href}".format(
                scheme="http://" if not urlparse(self.proxy).scheme else "",
                href=self.proxy,
            )
            self.public.proxies = self.private.proxies = {
                "http": proxy_href,
                "https": proxy_href,
            }
            return True
        self.public.proxies = self.private.proxies = {}
        return False
