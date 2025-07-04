import psycopg2
import pandas as pd
from typing import Optional, Dict, Any, List
from src.formula_one.logging import logger
from src.formula_one.entity.config_entity import DatabaseConfig

class DatabaseUtils:
    """Centralized database utilities for F1 data operations"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.logger = logger
    
    def connect_to_db(self):
        """Create database connection"""
        try:
            conn = psycopg2.connect(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.user,
                password=self.db_config.password
            )
            self.logger.info("Successfully connected to PostgreSQL database")
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load data from database table"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            if data:
                df = pd.DataFrame(data, columns=columns)
                return df
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading table {table_name}: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None):
        """Execute a database query"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            return cursor
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error executing query: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def execute_query_with_result(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query and return results"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """
        try:
            result = self.execute_query_with_result(query, (table_name,))
            return result[0][0] if result else False
        except Exception as e:
            self.logger.error(f"Error checking if table {table_name} exists: {e}")
            return False
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table"""
        query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        try:
            results = self.execute_query_with_result(query, (table_name,))
            return [row[0] for row in results]
        except Exception as e:
            self.logger.error(f"Error getting columns for table {table_name}: {e}")
            return []
    
    def get_table_structure(self, table_name: str) -> List[tuple]:
        """Get table structure (column_name, data_type, is_nullable)"""
        query = """
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position
        """
        try:
            return self.execute_query_with_result(query, (table_name,))
        except Exception as e:
            self.logger.error(f"Error getting table structure for {table_name}: {e}")
            return []