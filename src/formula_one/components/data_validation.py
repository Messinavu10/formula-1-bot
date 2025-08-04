import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.formula_one.logging import logger
from src.formula_one.components.data_ingestion import DatabaseIngestion
from src.formula_one.entity.config_entity import DataValidationConfig, DatabaseConfig
from formula_one.utils.validation_utils import ValidationUtils
from src.formula_one.components.base_component import BaseComponent

class RawDataValidator(BaseComponent):
    """Validates raw data before transformation"""
    
    def __init__(self, validation_config: DataValidationConfig, db_config: DatabaseConfig):
        super().__init__(validation_config, db_config)
        self.validation_results = {}
    
    def validate_all_tables(self) -> Dict[str, Any]:
        """Validate all raw tables"""
        logger.info("Starting raw data validation...")
        
        for table_name in self.config.tables_to_validate:
            logger.info(f"Validating raw table: {table_name}")
            try:
                df = self._load_table_data(table_name)
                if df is not None and not df.empty:
                    validation_result = self._validate_table(df, table_name)
                    self.validation_results[table_name] = validation_result
                else:
                    self.validation_results[table_name] = {
                        "passed": False,
                        "error": "Table is empty or could not be loaded"
                    }
            except Exception as e:
                logger.error(f"Error validating table {table_name}: {e}")
                self.validation_results[table_name] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Generate report
        if self.config.generate_validation_report:
            report_path = self.config.validation_report_dir / "raw_data_validation.json"  # Fixed filename
            ValidationUtils.generate_validation_report(
                self.validation_results, 
                report_path, 
                "raw"
            )
        
        return self.validation_results
    
    def _load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load data from database table"""
        conn = self.db_utils.connect_to_db()
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
            logger.error(f"Error loading table {table_name}: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    def _validate_table(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Validate a single table"""
        validation_result = {
            "passed": True,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": ValidationUtils.calculate_missing_percentage(df),
            "data_types": ValidationUtils.check_data_types(df),
            "unique_values": ValidationUtils.check_unique_values(df),
            "outliers": {},
            "issues": []
        }
        
        # Check missing values threshold
        for column, missing_pct in validation_result["missing_values"].items():
            if missing_pct > (self.config.missing_value_threshold * 100):
                validation_result["issues"].append(f"High missing values in {column}: {missing_pct}%")
                validation_result["passed"] = False
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            outlier_info = ValidationUtils.detect_outliers(df, column, self.config.outlier_threshold)
            validation_result["outliers"][column] = outlier_info
            
            if outlier_info["outlier_percentage"] > 10:  # More than 10% outliers
                validation_result["issues"].append(f"High outlier percentage in {column}: {outlier_info['outlier_percentage']}%")
        
        # Check for completely empty columns
        for column in df.columns:
            if df[column].isnull().all():
                validation_result["issues"].append(f"Column {column} is completely empty")
                validation_result["passed"] = False
        
        logger.info(f"Raw validation for {table_name}: {'PASSED' if validation_result['passed'] else 'FAILED'}")
        return validation_result


class TransformedDataValidator(BaseComponent):
    """Validates transformed data after transformation"""
    
    def __init__(self, validation_config: DataValidationConfig, db_config: DatabaseConfig):
        super().__init__(validation_config, db_config)
        self.validation_results = {}
    
    def validate_all_tables(self) -> Dict[str, Any]:
        """Validate all transformed tables"""
        logger.info("Starting transformed data validation...")
        
        for table_name in self.config.tables_to_validate:
            logger.info(f"Validating transformed table: {table_name}")
            try:
                df = self._load_table_data(f"{table_name}_transformed")
                if df is not None and not df.empty:
                    validation_result = self._validate_table(df, table_name)
                    self.validation_results[table_name] = validation_result
                else:
                    self.validation_results[table_name] = {
                        "passed": False,
                        "error": "Transformed table is empty or could not be loaded"
                    }
            except Exception as e:
                logger.error(f"Error validating transformed table {table_name}: {e}")
                self.validation_results[table_name] = {
                    "passed": False,
                    "error": str(e)
                }
        
        # Generate report
        if self.config.generate_validation_report:
            report_path = self.config.validation_report_dir / "transformed_data_validation.json"
            ValidationUtils.generate_validation_report(
                self.validation_results, 
                report_path, 
                "transformed"
            )
        
        return self.validation_results
    
    def _load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load data from database table"""
        conn = self.db_utils.connect_to_db()
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
            logger.error(f"Error loading table {table_name}: {e}")
            return None
        finally:
            cursor.close()
            conn.close()
    
    def _validate_table(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Validate a single transformed table"""
        validation_result = {
            "passed": True,
            "table_name": table_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": ValidationUtils.calculate_missing_percentage(df),
            "data_types": ValidationUtils.check_data_types(df),
            "unique_values": ValidationUtils.check_unique_values(df),
            "transformed_features": {},
            "issues": []
        }
        
        # Check for transformed features
        transformed_features = self._identify_transformed_features(df, table_name)
        validation_result["transformed_features"] = transformed_features
        
        # Validate transformed features
        for feature_type, features in transformed_features.items():
            for feature in features:
                if feature in df.columns:
                    feature_validation = self._validate_transformed_feature(df, feature, feature_type)
                    if not feature_validation["passed"]:
                        validation_result["issues"].extend(feature_validation["issues"])
                        validation_result["passed"] = False
        
        # Check missing values threshold
        for column, missing_pct in validation_result["missing_values"].items():
            if missing_pct > (self.config.missing_value_threshold * 100):
                validation_result["issues"].append(f"High missing values in {column}: {missing_pct}%")
                validation_result["passed"] = False
        
        logger.info(f"Transformed validation for {table_name}: {'PASSED' if validation_result['passed'] else 'FAILED'}")
        return validation_result
    
    def _identify_transformed_features(self, df: pd.DataFrame, table_name: str) -> Dict[str, List[str]]:
        """Identify transformed features in the table"""
        features = {
            "boolean_features": [],
            "encoded_features": [],
            "numeric_features": [],
            "statistical_features": []
        }
        
        for column in df.columns:
            if column.endswith('_encoded'):
                features["encoded_features"].append(column)
            elif column.startswith('is_') or column in ['had_incident', 'safety_car_lap', 'normal_pit_stop', 'long_pit_stop', 'penalty_pit_stop']:
                features["boolean_features"].append(column)
            elif any(suffix in column for suffix in ['_std', '_mean', '_deviation', '_consistency']):
                features["statistical_features"].append(column)
            elif any(suffix in column for suffix in ['_count', '_timing', '_duration', '_change', '_progression']):
                features["numeric_features"].append(column)
        
        return features
    
    def _validate_transformed_feature(self, df: pd.DataFrame, feature: str, feature_type: str) -> Dict[str, Any]:
        """Validate a specific transformed feature"""
        validation = {"passed": True, "issues": []}
        
        if feature_type == "boolean_features":
            # Check if boolean features have only True/False values
            unique_values = df[feature].unique()
            if not all(val in [True, False, 0, 1] for val in unique_values):
                validation["issues"].append(f"Boolean feature {feature} has non-boolean values: {unique_values}")
                validation["passed"] = False
        
        elif feature_type == "encoded_features":
            # Check if encoded features have reasonable ranges
            if df[feature].dtype in ['int64', 'float64']:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > 100:  # Assuming encoded values shouldn't be too large
                    validation["issues"].append(f"Encoded feature {feature} has unusually high values: max={max_val}")
                    validation["passed"] = False
        
        elif feature_type == "numeric_features":
            # Check for reasonable ranges in numeric features
            if df[feature].dtype in ['int64', 'float64']:
                outlier_info = ValidationUtils.detect_outliers(df, feature, self.config.outlier_threshold)
                if outlier_info["outlier_percentage"] > 20:  # More than 20% outliers
                    validation["issues"].append(f"High outlier percentage in {feature}: {outlier_info['outlier_percentage']}%")
                    validation["passed"] = False
        
        return validation


class DataValidation:
    """Main data validation class that combines raw and transformed validation"""
    
    def __init__(self, validation_config: DataValidationConfig, db_config: DatabaseConfig):
        self.config = validation_config
        self.db_config = db_config
        self.raw_validator = RawDataValidator(validation_config, db_config)
        self.transformed_validator = TransformedDataValidator(validation_config, db_config)
    
    def validate_raw_data(self) -> Dict[str, Any]:
        """Validate raw data before transformation"""
        if self.config.validate_raw_data:
            return self.raw_validator.validate_all_tables()
        else:
            logger.info("Raw data validation is disabled")
            return {}
    
    def validate_transformed_data(self) -> Dict[str, Any]:
        """Validate transformed data after transformation"""
        if self.config.validate_transformed_data:
            return self.transformed_validator.validate_all_tables()
        else:
            logger.info("Transformed data validation is disabled")
            return {}
    
    def validate_all(self) -> Dict[str, Any]:
        """Run both raw and transformed data validation"""
        logger.info("Starting comprehensive data validation...")
        
        results = {
            "raw_validation": self.validate_raw_data(),
            "transformed_validation": self.validate_transformed_data()
        }
        
        # Overall validation summary
        raw_passed = all(result.get("passed", False) for result in results["raw_validation"].values())
        transformed_passed = all(result.get("passed", False) for result in results["transformed_validation"].values())
        
        overall_passed = raw_passed and transformed_passed
        
        logger.info(f"Overall validation result: {'PASSED' if overall_passed else 'FAILED'}")
        
        return results