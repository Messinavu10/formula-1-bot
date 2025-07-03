import os
from src.formula_one.constants import *
from src.formula_one.utils.common import read_yaml, create_directories

from src.formula_one.entity.config_entity import DataIngestionConfig

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