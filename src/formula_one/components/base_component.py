from src.formula_one.logging import logger
from src.formula_one.utils.database_utils import DatabaseUtils
from src.formula_one.entity.config_entity import DatabaseConfig

class BaseComponent:
    """Base class for all components with shared functionality"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        self.config = config
        self.db_config = db_config
        self.db_utils = DatabaseUtils(db_config)
        self.logger = logger
    
    def load_table_data(self, table_name: str):
        """Load table data using centralized database utils"""
        return self.db_utils.load_table_data(table_name)
    
    def connect_to_db(self):
        """Get database connection using centralized database utils"""
        return self.db_utils.connect_to_db()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists using centralized database utils"""
        return self.db_utils.table_exists(table_name)
    
    def get_table_structure(self, table_name: str):
        """Get table structure using centralized database utils"""
        return self.db_utils.get_table_structure(table_name)