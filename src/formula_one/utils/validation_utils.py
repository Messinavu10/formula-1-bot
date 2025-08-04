import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.formula_one.logging import logger

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def calculate_missing_percentage(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing value percentage for each column"""
        missing_percentages = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_percentages[column] = round(missing_percentage, 2)
        return missing_percentages
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers in a numeric column using IQR method"""
        if df[column].dtype not in ['int64', 'float64']:
            return {"outlier_count": 0, "outlier_percentage": 0.0}
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        return {
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_percentage, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound)
        }
    
    @staticmethod
    def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Check data types of all columns"""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    @staticmethod
    def check_unique_values(df: pd.DataFrame) -> Dict[str, int]:
        """Check number of unique values in each column"""
        return {col: df[col].nunique() for col in df.columns}
    
    @staticmethod
    def generate_validation_report(
        validation_results: Dict[str, Any], 
        report_path: Path,
        validation_type: str = "raw"
    ) -> None:
        """Generate JSON validation report"""
        report_data = {
            "validation_type": validation_type,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tables": len(validation_results),
                "passed_tables": sum(1 for result in validation_results.values() if result.get("passed", False)),
                "failed_tables": sum(1 for result in validation_results.values() if not result.get("passed", False))
            },
            "detailed_results": validation_results
        }
        
        # Save JSON report (overwrites existing file)
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {report_path}")
        
        # Log summary
        logger.info(f"Validation Summary ({validation_type}):")
        logger.info(f"  Total tables: {report_data['summary']['total_tables']}")
        logger.info(f"  Passed: {report_data['summary']['passed_tables']}")
        logger.info(f"  Failed: {report_data['summary']['failed_tables']}")