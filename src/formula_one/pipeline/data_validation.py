# Add these imports at the top
from src.formula_one.logging import logger
import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.formula_one.entity.config_entity import DataValidationConfig, DatabaseConfig
from src.formula_one.config.configuration import ConfigurationManager
from src.formula_one.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    """Main training pipeline for data validation"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.validation_config = self.config_manager.get_data_validation_config()
        self.db_config = DatabaseConfig()
        self.data_validation = DataValidation(self.validation_config, self.db_config)
        self.logger = logger 
    
        self.logger.info(f"Validation configuration loaded - validate_raw: {self.validation_config.validate_raw_data}, validate_transformed: {self.validation_config.validate_transformed_data}")
    
    def initiate_raw_data_validation(self):
        """Initiate raw data validation pipeline"""
        self.logger.info("Starting raw data validation training pipeline")
        
        try:
            # Validate raw data
            raw_validation_results = self.data_validation.validate_raw_data()
            
            # Check if validation passed
            raw_passed = all(result.get("passed", False) for result in raw_validation_results.values())
            
            if raw_passed:
                self.logger.info("Raw data validation PASSED - all tables meet quality standards")
            else:
                self.logger.warning("Raw data validation FAILED - some tables have quality issues")
                self._log_validation_issues(raw_validation_results, "raw")
            
            self.logger.info("Raw data validation training pipeline completed!")
            return raw_validation_results
            
        except Exception as e:
            self.logger.error(f"Error in raw data validation pipeline: {e}")
            raise
    
    def initiate_transformed_data_validation(self):
        """Initiate transformed data validation pipeline"""
        self.logger.info("Starting transformed data validation training pipeline")
        
        try:
            # Validate transformed data
            transformed_validation_results = self.data_validation.validate_transformed_data()
            
            # Check if validation passed
            transformed_passed = all(result.get("passed", False) for result in transformed_validation_results.values())
            
            if transformed_passed:
                self.logger.info("Transformed data validation PASSED - all transformations successful")
            else:
                self.logger.warning("Transformed data validation FAILED - some transformations have issues")
                self._log_validation_issues(transformed_validation_results, "transformed")
            
            self.logger.info("Transformed data validation training pipeline completed!")
            return transformed_validation_results
            
        except Exception as e:
            self.logger.error(f"Error in transformed data validation pipeline: {e}")
            raise
    
    def initiate_comprehensive_validation(self):
        """Initiate comprehensive validation pipeline (both raw and transformed)"""
        self.logger.info("Starting comprehensive data validation training pipeline")
        
        try:
            # Run both validations
            all_validation_results = self.data_validation.validate_all()
            
            # Check overall results
            raw_passed = all(result.get("passed", False) for result in all_validation_results["raw_validation"].values())
            transformed_passed = all(result.get("passed", False) for result in all_validation_results["transformed_validation"].values())
            
            overall_passed = raw_passed and transformed_passed
            
            if overall_passed:
                self.logger.info("Comprehensive validation PASSED - all data meets quality standards")
            else:
                self.logger.warning("Comprehensive validation FAILED - some quality issues detected")
                if not raw_passed:
                    self._log_validation_issues(all_validation_results["raw_validation"], "raw")
                if not transformed_passed:
                    self._log_validation_issues(all_validation_results["transformed_validation"], "transformed")
            
            self.logger.info("Comprehensive data validation training pipeline completed!")
            return all_validation_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation pipeline: {e}")
            raise
    
    def _log_validation_issues(self, validation_results: Dict[str, Any], validation_type: str):
        """Log detailed validation issues for failed tables"""
        self.logger.warning(f"=== {validation_type.upper()} DATA VALIDATION ISSUES ===")
        
        for table_name, result in validation_results.items():
            if not result.get("passed", False):
                self.logger.warning(f"Table: {table_name}")
                
                if "error" in result:
                    self.logger.warning(f"  Error: {result['error']}")
                
                if "issues" in result and result["issues"]:
                    self.logger.warning(f"  Issues:")
                    for issue in result["issues"]:
                        self.logger.warning(f"    - {issue}")
                
                # Log missing values if high
                if "missing_values" in result:
                    high_missing = {col: pct for col, pct in result["missing_values"].items() 
                                  if pct > (self.validation_config.missing_value_threshold * 100)}
                    if high_missing:
                        self.logger.warning(f"  High missing values:")
                        for col, pct in high_missing.items():
                            self.logger.warning(f"    - {col}: {pct}%")
                
                # Log outlier information
                if "outliers" in result:
                    high_outliers = {col: info for col, info in result["outliers"].items() 
                                   if info.get("outlier_percentage", 0) > 10}
                    if high_outliers:
                        self.logger.warning(f"  High outlier percentages:")
                        for col, info in high_outliers.items():
                            self.logger.warning(f"    - {col}: {info['outlier_percentage']}%")
                
                # Log transformed feature issues
                if "transformed_features" in result:
                    features = result["transformed_features"]
                    if any(features.values()):
                        self.logger.warning(f"  Transformed features found:")
                        for feature_type, feature_list in features.items():
                            if feature_list:
                                self.logger.warning(f"    - {feature_type}: {feature_list}")
        
        self.logger.warning(f"=== END {validation_type.upper()} DATA VALIDATION ISSUES ===")
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of validation results"""
        summary = {
            "total_tables": len(validation_results),
            "passed_tables": 0,
            "failed_tables": 0,
            "failed_table_names": [],
            "overall_passed": False
        }
        
        for table_name, result in validation_results.items():
            if result.get("passed", False):
                summary["passed_tables"] += 1
            else:
                summary["failed_tables"] += 1
                summary["failed_table_names"].append(table_name)
        
        summary["overall_passed"] = summary["failed_tables"] == 0
        
        return summary