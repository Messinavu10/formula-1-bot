# Add these imports at the top
from src.formula_one.logging import logger
import requests
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from urllib.parse import urlencode
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.formula_one.entity.config_entity import DataIngestionConfig, DatabaseConfig
from src.formula_one.config.configuration import ConfigurationManager
from src.formula_one.components.data_ingestion import OpenF1APIClient, DataIngestion, DatabaseIngestion

class DataIngestionTrainingPipeline:
    """Main training pipeline for data ingestion"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.data_ingestion_config = self.config_manager.get_data_ingestion_config()
        self.db_config = DatabaseConfig()
        self.api_client = OpenF1APIClient(self.data_ingestion_config)
        self.data_ingestion = DataIngestion(self.data_ingestion_config, self.api_client)
        self.db_ingestion = DatabaseIngestion(self.data_ingestion_config, self.db_config, self.api_client)
        self.logger = logger 
    
        self.logger.info(f"Configuration loaded - max_retries: {self.data_ingestion_config.max_retries} (type: {type(self.data_ingestion_config.max_retries)})")
    
    def initiate_data_ingestion(self):
        """Initiate the complete data ingestion pipeline"""
        self.logger.info("Starting data ingestion training pipeline")
        
        try:
            # Create database tables
            self.db_ingestion.create_tables()
            
            # Discover available 2025 meetings
            meetings = self._discover_2025_meetings()
            
            # Ingest data for each meeting
            for meeting in meetings:
                meeting_key = str(meeting.get('meeting_key'))
                meeting_name = meeting.get('meeting_name')
                
                self.logger.info(f"Ingesting {meeting_name} (Key: {meeting_key})")
                
                # Ingest data for this meeting
                meeting_data = self.data_ingestion.ingest_meeting_data(meeting_key)
                
                # Insert into database
                if meeting_data:
                    self.db_ingestion.insert_meeting_data(meeting_data)
                    self.logger.info(f"{meeting_name} data successfully ingested!")
                else:
                    self.logger.warning(f"No data found for {meeting_name}")
            
            self.logger.info("Data ingestion training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in data ingestion pipeline: {e}")
            raise
    
    def _discover_2025_meetings(self):
        """Discover all available 2025 meetings"""
        meetings = self.api_client.get_meetings()
        meetings_2025 = [m for m in meetings if m.get('year') == 2025]
        meetings_2025.sort(key=lambda x: x.get('date_start', ''))
        
        self.logger.info(f"Found {len(meetings_2025)} meetings in 2025")
        return meetings_2025