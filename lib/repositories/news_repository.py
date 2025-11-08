"""新聞資料庫存取層"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime


class INewsRepository(ABC):
    """新聞資料庫存取介面"""
    
    @abstractmethod
    def get_random_news(self, 
                       date_filter: str,
                       exclude_categories: list[str] = None) -> Optional[Dict[str, Any]]:
        """獲取隨機新聞
        
        Args:
            date_filter: 日期篩選條件
            exclude_categories: 要排除的類別列表
            
        Returns:
            包含 title, keyword 的字典，如果沒有則返回 None
        """
        pass


class NewsRepository(INewsRepository):
    """新聞資料庫存取實現"""
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.default_exclude_categories = [
            '政治', '兩岸', '美股雷達', 'A股港股', 
            '財經雲', '股市', '台股新聞', '理財', '股票',
            '大陸房市', '美股', '港股', 'A股', '債券',
            '新股上市', '指數', '外資', '法人', '主力',
            '台股盤勢', '大陸政經', '幣圈', '美股預估',
            'ETF', '外匯'
        ]
    
    def get_random_news(self, 
                       date_filter: str,
                       exclude_categories: list[str] = None) -> Optional[Dict[str, Any]]:
        """獲取隨機新聞"""
        import random
        
        if exclude_categories is None:
            exclude_categories = self.default_exclude_categories
            
        engine = self.db_connection.engine
        
        # 讀取新聞資料
        df = pd.read_sql_query(
            f"""
            SELECT title, keyword, created_at, category 
            FROM news_ch.news 
            WHERE category NOT IN {tuple(exclude_categories)} AND keyword != '' 
            AND created_at >= '{date_filter}'
            ORDER BY id DESC 
            LIMIT 10000
            """, 
            engine
        )
        
        if len(df) == 0:
            return None
        
        # 隨機選擇一則新聞
        choose_index = random.choice(range(0, len(df)))
        
        return {
            'title': df.loc[choose_index, 'title'],
            'keyword': df.loc[choose_index, 'keyword']
        } 