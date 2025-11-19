from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
import os
import mysql.connector
from mysql.connector.connection import MySQLConnection
from mysql.connector import Error as MySQLError

# -----------------------------------------------------------------------------
# Configuration via environment variables
# -----------------------------------------------------------------------------
DB_CONFIG: Dict[str, Any] = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "autocommit": True,  # simple microservice CRUD pattern
}

class AbstractBaseMySQLService(ABC):
    """
    Base class that manages a MySQL connection and enforces CRUD interface.
    """

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self._db_config = db_config or DB_CONFIG
        self._connection: Optional[MySQLConnection] = None

    def connect(self) -> None:
        if self._connection is None or not self._connection.is_connected():
            try:
                host = self._db_config.get("host")
                print(f"[DB] Connecting to MySQL at {host} ...")
                self._connection = mysql.connector.connect(**self._db_config)
                print("[DB] Connected.")
            except MySQLError as err:
                print(f"[DB] Connection failed: {err}")
                self._connection = None
                raise

    def get_connection(self) -> MySQLConnection:
        if self._connection is None or not self._connection.is_connected():
            self.connect()
        return self._connection  # type: ignore

    def close_connection(self) -> None:
        if self._connection and self._connection.is_connected():
            self._connection.close()
            self._connection = None
            print("[DB] Connection closed.")

    # --- CRUD interface (override in subclasses) ---
    @abstractmethod
    def create(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def retrieve(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def delete(self, *args, **kwargs) -> Any:
        pass
