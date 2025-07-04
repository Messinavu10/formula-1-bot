#!/usr/bin/env python3
"""
Test script for the modularized data transformation pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.formula_one.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.formula_one.logging import logger

def test_data_transformation():
    """Test the data transformation pipeline"""
    try:
        logger.info("Starting data transformation test...")
        
        # Initialize the pipeline
        pipeline = DataTransformationTrainingPipeline()
        
        # Run the transformation
        transformed_data = pipeline.initiate_data_transformation()
        
        # Get summary
        summary = pipeline.get_transformation_summary(transformed_data)
        
        logger.info("Data transformation test completed successfully!")
        logger.info(f"Summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data transformation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_data_transformation()
    if success:
        print("✅ Data transformation test PASSED")
    else:
        print("❌ Data transformation test FAILED")
        sys.exit(1) 