import pymysql
import psycopg2
import pyodbc
from sqlalchemy import create_engine
from typing import Dict, Optional, Any
from contextlib import contextmanager
import logging
from threading import Lock
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

class DatabaseConnectionPool:
    _instance = None
    _lock = Lock()
    _connections: Dict[str, 'DatabaseConnection'] = {}
    _connection_params: Dict[str, Dict[str, Any]] = {}
    _db_names: Dict[str, str] = {}  # 保存每種資料庫類型對應的資料庫名稱

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_connection(self, db_type: str, **kwargs) -> 'DatabaseConnection':
        """
        獲取資料庫連線，如果連線不存在則創建新的連線
        
        使用範例:
        # 第一次使用時需要初始化
        db_pool = DatabaseConnectionPool()
        db_pool.initialize('mysql', host='localhost', port=3306, 
                          user='user', password='pass', db_name='test')
        
        # 之後只需要指定 db_type 即可獲取連線
        db = db_pool.get_connection('mysql')
        db.cursor.execute('SELECT * FROM table')
        results = db.cursor.fetchall()
        """
        db_type = db_type.lower()
        
        # 如果有新的參數，使用新的參數
        if 'db_name' in kwargs:
            self._db_names[db_type] = kwargs['db_name']
        
        # 獲取資料庫名稱
        db_name = self._db_names.get(db_type, '')
        if not db_name and not kwargs:
            raise ValueError(f"No connection parameters found for {db_type}. Please initialize first.")
            
        connection_key = f"{db_type}_{db_name}"

        # 如果連線不存在，檢查是否有保存的參數
        if connection_key not in self._connections:
            if not kwargs and connection_key in self._connection_params:
                # 使用保存的參數創建連線
                kwargs = self._connection_params[connection_key]
            elif not kwargs:
                raise ValueError(f"No connection parameters found for {db_type}. Please initialize first.")
            
            # 創建新連線
            if db_type == 'mysql':
                self._connections[connection_key] = MySQLConnection(
                    host=kwargs['host'],
                    port=kwargs['port'],
                    user=kwargs['user'],
                    password=kwargs['password'],
                    db_name=kwargs['db_name']
                )
            elif db_type == 'postgresql':
                self._connections[connection_key] = PostgreSQLConnection(
                    host=kwargs['host'],
                    port=kwargs['port'],
                    user=kwargs['user'],
                    password=kwargs['password'],
                    db_name=kwargs['db_name']
                )
            elif db_type == 'mssql':
                self._connections[connection_key] = MSSQLConnection(
                    host=kwargs['host'],
                    port=kwargs['port'],
                    user=kwargs['user'],
                    password=kwargs['password'],
                    db_name=kwargs['db_name']
                )
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # 保存連線參數
            if kwargs:
                self._connection_params[connection_key] = kwargs
                self._db_names[db_type] = kwargs['db_name']
            
        return self._connections[connection_key]

    def initialize(self, db_type: str, **kwargs) -> None:
        """
        初始化資料庫連線
        
        使用範例:
        db_pool.initialize('mysql', 
                          host='localhost',
                          port=3306,
                          user='user',
                          password='pass',
                          db_name='test')
        """
        if not all(key in kwargs for key in ['host', 'port', 'user', 'password', 'db_name']):
            raise ValueError("Missing required connection parameters. Required: host, port, user, password, db_name")
        
        self.get_connection(db_type, **kwargs)

    def close_all(self) -> None:
        """
        關閉所有資料庫連線
        """
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    def close_connection(self, db_type: str, db_name: str) -> None:
        """
        關閉指定的資料庫連線
        """
        connection_key = f"{db_type}_{db_name}"
        if connection_key in self._connections:
            self._connections[connection_key].close()
            del self._connections[connection_key]

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