from src.formula_one.logging import logger 
from src.formula_one.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.formula_one.pipeline.data_validation import DataValidationTrainingPipeline
from src.formula_one.pipeline.data_transformation import DataTransformationTrainingPipeline
from src.formula_one.pipeline.mcp_pipeline import MCPTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME_2 = "Data Validation (Raw) stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME_2} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   raw_validation_results = data_validation.initiate_raw_data_validation()
   logger.info(f">>>>>> stage {STAGE_NAME_2} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME_3 = "Data Transformation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME_3} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   transformed_data = data_transformation.initiate_data_transformation()
   logger.info(f">>>>>> stage {STAGE_NAME_3} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME_4 = "Data Validation (Transformed) stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME_4} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   transformed_validation_results = data_validation.initiate_transformed_data_validation()
   logger.info(f">>>>>> stage {STAGE_NAME_4} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME_5 = "MCP System stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME_5} started <<<<<<") 
   mcp_pipeline = MCPTrainingPipeline()
   mcp_pipeline.initiate_mcp_system()
   logger.info(f">>>>>> stage {STAGE_NAME_5} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

logger.info("All pipeline stages completed successfully!")