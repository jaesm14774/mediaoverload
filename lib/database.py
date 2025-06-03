import pymysql
import psycopg2
import pyodbc
from sqlalchemy import create_engine
from typing import Dict, Optional, Any
from contextlib import contextmanager
import logging
from threading import Lock
from urllib.parse import quote_plus
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    _instance = None
    _lock = Lock()
    _connections: Dict[str, 'DatabaseConnection'] = {}
    _connection_params: Dict[str, Dict[str, Any]] = {}
    _db_names: Dict[str, str] = {}  # 保存每種資料庫類型對應的資料庫名稱
    _last_used: Dict[str, datetime] = {}  # 記錄每個連接最後使用時間

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _check_connection_health(self, connection: 'DatabaseConnection', connection_key: str) -> bool:
        """檢查連接是否健康"""
        try:
            if connection.conn and hasattr(connection.conn, 'ping'):
                connection.conn.ping(reconnect=True)
            return True
        except Exception as e:
            logger.error(f"Connection health check failed for {connection_key}: {str(e)}")
            return False

    def _create_new_connection(self, db_type: str, kwargs: Dict[str, Any]) -> 'DatabaseConnection':
        """創建新的資料庫連接"""
        if db_type == 'mysql':
            return MySQLConnection(
                host=kwargs['host'],
                port=kwargs['port'],
                user=kwargs['user'],
                password=kwargs['password'],
                db_name=kwargs['db_name']
            )
        elif db_type == 'postgresql':
            return PostgreSQLConnection(
                host=kwargs['host'],
                port=kwargs['port'],
                user=kwargs['user'],
                password=kwargs['password'],
                db_name=kwargs['db_name']
            )
        elif db_type == 'mssql':
            return MSSQLConnection(
                host=kwargs['host'],
                port=kwargs['port'],
                user=kwargs['user'],
                password=kwargs['password'],
                db_name=kwargs['db_name']
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def get_connection(self, db_type: str, **kwargs) -> 'DatabaseConnection':
        """獲取資料庫連線，如果連線不存在或已斷開則創建新的連線"""
        db_type = db_type.lower()
        
        # 如果有新的參數，使用新的參數
        if 'db_name' in kwargs:
            self._db_names[db_type] = kwargs['db_name']
        
        # 獲取資料庫名稱
        db_name = self._db_names.get(db_type, '')
        if not db_name and not kwargs:
            raise ValueError(f"No connection parameters found for {db_type}. Please initialize first.")
            
        connection_key = f"{db_type}_{db_name}"

        with self._lock:
            # 檢查現有連接
            if connection_key in self._connections:
                connection = self._connections[connection_key]
                # 檢查連接健康狀態
                if self._check_connection_health(connection, connection_key):
                    self._last_used[connection_key] = datetime.now()
                    return connection
                else:
                    # 如果連接不健康，關閉並移除
                    connection.close()
                    del self._connections[connection_key]
                    if connection_key in self._last_used:
                        del self._last_used[connection_key]

            # 創建新連接
            if not kwargs and connection_key in self._connection_params:
                kwargs = self._connection_params[connection_key]
            elif not kwargs:
                raise ValueError(f"No connection parameters found for {db_type}. Please initialize first.")

            try:
                connection = self._create_new_connection(db_type, kwargs)
                self._connections[connection_key] = connection
                self._last_used[connection_key] = datetime.now()
                
                # 保存連線參數
                if kwargs:
                    self._connection_params[connection_key] = kwargs
                    self._db_names[db_type] = kwargs['db_name']
                
                return connection
            except Exception as e:
                logger.error(f"Failed to create new connection for {connection_key}: {str(e)}")
                raise

    def initialize(self, db_type: str, **kwargs) -> None:
        """初始化資料庫連線"""
        if not all(key in kwargs for key in ['host', 'port', 'user', 'password', 'db_name']):
            raise ValueError("Missing required connection parameters. Required: host, port, user, password, db_name")
        
        self.get_connection(db_type, **kwargs)

    def close_all(self) -> None:
        """關閉所有資料庫連線"""
        with self._lock:
            for conn in self._connections.values():
                conn.close()
            self._connections.clear()
            self._last_used.clear()

    def close_connection(self, db_type: str, db_name: str) -> None:
        """關閉指定的資料庫連線"""
        connection_key = f"{db_type}_{db_name}"
        with self._lock:
            if connection_key in self._connections:
                self._connections[connection_key].close()
                del self._connections[connection_key]
                if connection_key in self._last_used:
                    del self._last_used[connection_key]

    def cleanup_idle_connections(self, max_idle_time: int = 3600) -> None:
        """清理閒置的連接"""
        current_time = datetime.now()
        with self._lock:
            for connection_key, last_used in list(self._last_used.items()):
                if (current_time - last_used).total_seconds() > max_idle_time:
                    self._connections[connection_key].close()
                    del self._connections[connection_key]
                    del self._last_used[connection_key]

class DatabaseConnection:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.engine = None

    def close(self):
        """關閉資料庫連線"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.dispose()

class MySQLConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, user: str, password: str, db_name: str):
        super().__init__()
        try:
            # 建立 MySQL 連線
            self.conn = pymysql.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=db_name,
                charset='utf8mb4'
            )
            self.cursor = self.conn.cursor()
            
            # 建立 SQLAlchemy engine
            self.engine = create_engine(
                f'mysql+pymysql://{user}:{quote_plus(password)}@{host}:{port}/{db_name}?charset=utf8mb4'
            )
            logger.info(f"Successfully connected to MySQL database: {db_name}")
        except Exception as e:
            logger.error(f"Error connecting to MySQL: {str(e)}")
            raise

class PostgreSQLConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, user: str, password: str, db_name: str):
        super().__init__()
        try:
            # 建立 PostgreSQL 連線
            self.conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                dbname=db_name
            )
            self.cursor = self.conn.cursor()
            
            # 建立 SQLAlchemy engine
            self.engine = create_engine(
                f'postgresql://{user}:{quote_plus(password)}@{host}:{port}/{db_name}'
            )
            logger.info(f"Successfully connected to PostgreSQL database: {db_name}")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            raise

class MSSQLConnection(DatabaseConnection):
    def __init__(self, host: str, port: int, user: str, password: str, db_name: str):
        super().__init__()
        try:
            # 建立 MSSQL 連線
            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={host},{port};"
                f"DATABASE={db_name};"
                f"UID={user};"
                f"PWD={password};"
                f"TrustServerCertificate=yes"
            )
            self.conn = pyodbc.connect(conn_str)
            self.cursor = self.conn.cursor()
            
            # 建立 SQLAlchemy engine
            self.engine = create_engine(
                f'mssql+pyodbc://{user}:{quote_plus(password)}@{host}:{port}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes'
            )
            logger.info(f"Successfully connected to MSSQL database: {db_name}")
        except Exception as e:
            logger.error(f"Error connecting to MSSQL: {str(e)}")
            raise

# 創建全局連線池實例
db_pool = DatabaseConnectionPool() 