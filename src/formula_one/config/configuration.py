import os
from src.formula_one.constants import *
from src.formula_one.utils.common import read_yaml, create_directories

from src.formula_one.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig

class ConfigurationManager:
    """Manages configuration loading from YAML files"""
    
    def __init__(self, config_file_path: str = "config/config.yaml"):
        self.config_file_path = Path(config_file_path)
        self.config = read_yaml(self.config_file_path)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        config_data = self.config.get('data_ingestion', {})
        api_config = config_data.get('api', {})
        
        return DataIngestionConfig(
            # Convert strings to proper types
            source_url=str(api_config.get('base_url', 'https://api.openf1.org/v1')),
            api_timeout=int(api_config.get('timeout', 30)),
            max_retries=int(api_config.get('max_retries', 3)),
            batch_size=int(api_config.get('batch_size', 1000)),
            target_meetings=config_data.get('target_meetings', [{"meeting_key": "latest"}]),
            sessions_to_ingest=config_data.get('sessions_to_ingest', [
                "Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"
            ]),
            endpoints=config_data.get('endpoints', [
                "meetings", "sessions", "drivers", "laps", "pit", "stints", 
                "position", "intervals", "team_radio", "race_control", "weather"
            ])
        )
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration"""
        config_data = self.config.get('data_validation', {})
        
        return DataValidationConfig(
            root_dir=Path(config_data.get('root_dir', 'artifacts/data_validation')),
            validation_report_dir=Path(config_data.get('validation_report_dir', 'artifacts/data_validation')),
            data_quality_threshold=float(config_data.get('data_quality_threshold', 0.95)),
            missing_value_threshold=float(config_data.get('missing_value_threshold', 0.1)),
            outlier_threshold=float(config_data.get('outlier_threshold', 3.0)),
            validate_raw_data=config_data.get('validate_raw_data', True),
            validate_transformed_data=config_data.get('validate_transformed_data', True),
            generate_validation_report=config_data.get('generate_validation_report', True),
            save_validation_results=config_data.get('save_validation_results', True),
            validation_report_format=config_data.get('validation_report_format', 'json'),
            tables_to_validate=config_data.get('tables_to_validate', [
                "meetings", "sessions", "drivers", "laps", 
                "pit_stops", "stints", "positions", "intervals", 
                "weather"
            ])
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration"""
        config_data = self.config.get('data_transformation', {})
        
        return DataTransformationConfig(
            missing_value_strategy=config_data.get('missing_value_strategy', 'impute'),
            outlier_strategy=config_data.get('outlier_strategy', 'cap'),
            outlier_threshold=float(config_data.get('outlier_threshold', 3.0)),
            create_tire_features=config_data.get('create_tire_features', True),
            create_lap_features=config_data.get('create_lap_features', True),
            create_weather_features=config_data.get('create_weather_features', True),
            create_driver_features=config_data.get('create_driver_features', True),
            numeric_columns=config_data.get('numeric_columns', [
                "lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3",
                "pit_duration", "lap_start", "lap_end", "tyre_age_at_start",
                "position", "gap_to_leader", "interval",
                "air_temperature", "track_temperature", "humidity"
            ]),
            categorical_columns=config_data.get('categorical_columns', [
                "compound", "flag", "category", "scope"
            ]),
            datetime_columns=config_data.get('datetime_columns', [
                "date_start", "date_end", "date"
            ]),
            tables_to_transform=config_data.get('tables_to_transform', [
                "sessions", "drivers", "laps", 
                "pit_stops", "stints", "positions", "intervals", 
                "weather"
            ])
        )