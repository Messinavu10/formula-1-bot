from src.formula_one.logging import logger
import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.formula_one.entity.config_entity import DataTransformationConfig, DatabaseConfig
from src.formula_one.config.configuration import ConfigurationManager
from src.formula_one.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    """Main training pipeline for data transformation"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.transformation_config = self.config_manager.get_data_transformation_config()
        self.db_config = DatabaseConfig()
        self.data_transformation = DataTransformation(self.transformation_config, self.db_config)
        self.logger = logger 
    
        self.logger.info(f"Transformation configuration loaded - tables to transform: {self.transformation_config.tables_to_transform}")
    
    def initiate_data_transformation(self):
        """Initiate the complete data transformation pipeline"""
        self.logger.info("Starting data transformation training pipeline")
        
        try:
            # Transform all data
            transformed_data = self.data_transformation.transform_all_data()
            
            # Log transformation summary
            self._log_transformation_summary(transformed_data)
            
            self.logger.info("Data transformation training pipeline completed successfully!")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error in data transformation pipeline: {e}")
            raise
    
    def _log_transformation_summary(self, transformed_data: Dict[str, pd.DataFrame]):
        """Log a summary of the transformation results"""
        self.logger.info("=== DATA TRANSFORMATION SUMMARY ===")
        
        total_tables = len(transformed_data)
        total_rows = sum(len(df) for df in transformed_data.values())
        
        self.logger.info(f"Total tables transformed: {total_tables}")
        self.logger.info(f"Total rows processed: {total_rows}")
        
        for table_name, df in transformed_data.items():
            self.logger.info(f"Table: {table_name}")
            self.logger.info(f"  Rows: {len(df)}")
            self.logger.info(f"  Columns: {len(df.columns)}")
            
            # Log encoded columns
            encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
            if encoded_cols:
                self.logger.info(f"  Encoded columns: {encoded_cols}")
            
            # Log feature columns
            feature_cols = [col for col in df.columns if any(feature in col for feature in 
                           ['std', 'mean', 'deviation', 'consistency', 'count', 'timing', 
                            'duration', 'change', 'progression', 'delta', 'severity'])]
            if feature_cols:
                self.logger.info(f"  Feature columns: {feature_cols}")
            
            # Log boolean columns
            boolean_cols = [col for col in df.columns if col.startswith('is_') or 
                          col.endswith('_improved') or col.endswith('_declined') or
                          col.endswith('_incident') or col.endswith('_car') or
                          col.endswith('_stop') or col.endswith('_weather')]
            if boolean_cols:
                self.logger.info(f"  Boolean columns: {boolean_cols}")
        
        self.logger.info("=== END DATA TRANSFORMATION SUMMARY ===")
    
    def get_transformation_summary(self, transformed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get a summary of transformation results"""
        summary = {
            "total_tables": len(transformed_data),
            "total_rows": sum(len(df) for df in transformed_data.values()),
            "table_details": {},
            "overall_success": True
        }
        
        for table_name, df in transformed_data.items():
            table_summary = {
                "rows": len(df),
                "columns": len(df.columns),
                "encoded_columns": [col for col in df.columns if col.endswith('_encoded')],
                "feature_columns": [col for col in df.columns if any(feature in col for feature in 
                                  ['std', 'mean', 'deviation', 'consistency', 'count', 'timing', 
                                   'duration', 'change', 'progression', 'delta', 'severity'])],
                "boolean_columns": [col for col in df.columns if col.startswith('is_') or 
                                  col.endswith('_improved') or col.endswith('_declined') or
                                  col.endswith('_incident') or col.endswith('_car') or
                                  col.endswith('_stop') or col.endswith('_weather')]
            }
            summary["table_details"][table_name] = table_summary
        
        return summary 