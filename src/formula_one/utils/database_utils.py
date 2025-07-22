import psycopg2
import psycopg2.extras
import psycopg2.pool
import pandas as pd
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from src.formula_one.logging import logger
from src.formula_one.entity.config_entity import DatabaseConfig
from sqlalchemy import text, create_engine

class DatabaseUtils:
    """Centralized database utilities for F1 data operations"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.logger = logger
        self.connection_pool = None
        self._initialize_pool()
    
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
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.user,
                password=self.db_config.password
            )
            self.logger.info("Database connection pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with context manager"""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
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
    
    def execute_mcp_query(self, query: str, params: dict = None):
        """Execute a query for MCP tools with logging"""
        try:
            logger.info(f"Executing MCP query: {query[:100]}...")
            conn = self.connect_to_db()
            cursor = conn.cursor()
            if params:
                psycopg2.extras.execute_values(cursor, query, list(params.values()), page_size=1000)
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
            logger.info(f"MCP query executed successfully, returned {len(rows)} rows")
            return rows
        except Exception as e:
            logger.error(f"MCP query failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_schema_info(self):
        """Get comprehensive database schema information"""
        schema_info = {}
        try:
            conn = self.connect_to_db()
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                    ORDER BY ordinal_position
                """, (table,))
                columns = [{"name": row[0], "type": row[1], "nullable": row[2]} for row in cursor.fetchall()]
                schema_info[table] = {"columns": columns}
                cursor.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
                """, (table,))
                schema_info[table]["primary_keys"] = [row[0] for row in cursor.fetchall()]
            self.logger.info(f"Schema loaded for {len(tables)} tables")
            return schema_info
        except Exception as e:
            self.logger.error(f"Error loading schema: {e}")
            return {}