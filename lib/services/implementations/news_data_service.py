"""新聞資料服務實現"""
import random
from typing import Dict, Any, Optional
import pandas as pd
from lib.services.interfaces.news_data_service import INewsDataService


class NewsDataService(INewsDataService):
    """新聞資料服務實現"""
    
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
        if exclude_categories is None:
            exclude_categories = self.default_exclude_categories
            
        engine = self.db_connection.engine
        
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
        
        choose_index = random.choice(range(0, len(df)))
        
        return {
            'title': df.loc[choose_index, 'title'],
            'keyword': df.loc[choose_index, 'keyword']
        }

