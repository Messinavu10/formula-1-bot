import requests
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from urllib.parse import urlencode
from typing import Optional, Dict, Any, List
from pathlib import Path
from src.formula_one.logging import logger
from src.formula_one.entity.config_entity import DataIngestionConfig, DatabaseConfig
from src.formula_one.utils.database_utils import DatabaseUtils

class OpenF1APIClient:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.api_timeout
        self.logger = logger 
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Make API request with retry logic and rate limiting"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        url = f"{self.config.source_url}/{endpoint}"
        
        if params:
            url += f"?{urlencode(params)}"
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Making request to: {url}")
                response = self.session.get(url)
                self.last_request_time = time.time()
                
                response.raise_for_status()
                
                data = response.json()
                self.logger.info(f"Successfully retrieved {len(data)} records from {endpoint}")
                return data
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {endpoint}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def get_meetings(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get meetings/race weekends"""
        params = {"year": year} if year else {}
        return self._make_request("meetings", params)
    
    def get_sessions(self, meeting_key: str) -> List[Dict[str, Any]]:
        """Get sessions for a specific meeting"""
        params = {"meeting_key": meeting_key}
        return self._make_request("sessions", params)
    
    def get_drivers(self, session_key: str) -> List[Dict[str, Any]]:
        """Get drivers for a specific session"""
        params = {"session_key": session_key}
        return self._make_request("drivers", params)
    
    def get_laps(self, session_key: str) -> List[Dict[str, Any]]:
        """Get lap data for a specific session"""
        params = {"session_key": session_key}
        return self._make_request("laps", params)
    
    def get_pit_stops(self, session_key: str) -> List[Dict[str, Any]]:
        """Get pit stop data"""
        params = {"session_key": session_key}
        return self._make_request("pit", params)
    
    def get_stints(self, session_key: str) -> List[Dict[str, Any]]:
        """Get stint data"""
        params = {"session_key": session_key}
        return self._make_request("stints", params)
    
    def get_positions(self, session_key: str) -> List[Dict[str, Any]]:
        """Get position data"""
        params = {"session_key": session_key}
        return self._make_request("position", params)
    
    def get_intervals(self, session_key: str) -> List[Dict[str, Any]]:
        """Get interval data"""
        params = {"session_key": session_key}
        return self._make_request("intervals", params)
      
    def get_race_control(self, session_key: str) -> List[Dict[str, Any]]:
        """Get race control data"""
        params = {"session_key": session_key}
        return self._make_request("race_control", params)
    
    def get_weather(self, session_key: str) -> List[Dict[str, Any]]:
        """Get weather data"""
        params = {"session_key": session_key}
        return self._make_request("weather", params)

class DatabaseIngestion:
    """Handles database operations for F1 data"""
    
    def __init__(self, data_config: DataIngestionConfig, db_config: DatabaseConfig, api_client: OpenF1APIClient):
        self.data_config = data_config
        self.db_config = db_config
        self.api_client = api_client
        self.db_utils = DatabaseUtils(db_config)
        self.logger = logger
    
    def connect_to_db(self):
        """Create database connection"""
        return self.db_utils.connect_to_db()
    
    def _parse_interval_value(self, value):
        """Parse interval values like '+4 LAPS' or '12.345' to float"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            if value == "+1 LAP" or value == "+2 LAPS" or value == "+3 LAPS" or value == "+4 LAPS":
                # Convert "+X LAPS" to a large negative number to represent lapped
                return -999.0  # Use -999 to represent lapped
            try:
                return float(value)
            except ValueError:
                self.logger.warning(f"Could not parse interval value: {value}, setting to None")
                return None
        return None
    
    # Update the create_tables method with optimized ML-focused schema
    def create_tables(self):
        """Create database tables optimized for ML models"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        # Drop existing tables to recreate with optimized schema
        cursor.execute("DROP TABLE IF EXISTS weather CASCADE")
        cursor.execute("DROP TABLE IF EXISTS race_control CASCADE")
        cursor.execute("DROP TABLE IF EXISTS intervals CASCADE")
        cursor.execute("DROP TABLE IF EXISTS stints CASCADE")
        cursor.execute("DROP TABLE IF EXISTS pit_stops CASCADE")
        cursor.execute("DROP TABLE IF EXISTS positions CASCADE")
        cursor.execute("DROP TABLE IF EXISTS laps CASCADE")
        cursor.execute("DROP TABLE IF EXISTS drivers CASCADE")
        cursor.execute("DROP TABLE IF EXISTS sessions CASCADE")
        cursor.execute("DROP TABLE IF EXISTS meetings CASCADE")
        
        # Core tables with essential fields only
        cursor.execute("""
            CREATE TABLE meetings (
                meeting_key INTEGER PRIMARY KEY,
                meeting_name VARCHAR(255),
                country_name VARCHAR(100),
                circuit_short_name VARCHAR(100),
                date_start TIMESTAMP,
                year INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE sessions (
                session_key INTEGER PRIMARY KEY,
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                session_name VARCHAR(100),
                session_type VARCHAR(50),
                date_start TIMESTAMP,
                date_end TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE drivers (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                full_name VARCHAR(100),
                team_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE laps (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                lap_number INTEGER,
                lap_duration FLOAT,
                duration_sector_1 FLOAT,
                duration_sector_2 FLOAT,
                duration_sector_3 FLOAT,
                is_pit_out_lap BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE pit_stops (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                lap_number INTEGER,
                pit_duration FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE stints (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                compound VARCHAR(20),
                lap_start INTEGER,
                lap_end INTEGER,
                tyre_age_at_start INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE positions (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                position INTEGER,
                date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE weather (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                air_temperature FLOAT,
                track_temperature FLOAT,
                humidity FLOAT,
                rainfall BOOLEAN,
                date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE intervals (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                gap_to_leader FLOAT,  -- Time gap to race leader in seconds
                interval FLOAT,       -- Time gap to car ahead in seconds
                date TIMESTAMP,       -- UTC date and time
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE race_control (
                id SERIAL PRIMARY KEY,
                session_key INTEGER REFERENCES sessions(session_key),
                meeting_key INTEGER REFERENCES meetings(meeting_key),
                driver_number INTEGER,
                category VARCHAR(50),
                flag VARCHAR(20),
                lap_number INTEGER,
                message TEXT,
                scope VARCHAR(20),
                sector INTEGER,
                date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        self.logger.info("Optimized database tables created successfully")
    
    def insert_meeting_data(self, meeting_data: Dict[str, Any]):
        """Insert meeting data into database with optimized schema"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # Insert meeting info - OPTIMIZED FIELDS ONLY
            meeting_info = meeting_data["meeting_info"]
            cursor.execute("""
                INSERT INTO meetings (
                    meeting_key, meeting_name, country_name, 
                    circuit_short_name, date_start, year
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (meeting_key) DO NOTHING
            """, (
                meeting_info.get('meeting_key'),
                meeting_info.get('meeting_name'),
                meeting_info.get('country_name'),
                meeting_info.get('circuit_short_name'),
                meeting_info.get('date_start'),
                meeting_info.get('year')
            ))
            
            # Get the original sessions list
            sessions_list = self.api_client.get_sessions(str(meeting_info.get('meeting_key')))
            
            # Insert sessions - OPTIMIZED FIELDS ONLY
            for session_name, session_data in meeting_data["sessions"].items():
                # Find the session info from the original sessions list
                session_info = next((s for s in sessions_list if s.get('session_name') == session_name), None)
                
                if session_info:
                    cursor.execute("""
                        INSERT INTO sessions (
                            session_key, meeting_key, session_name, session_type,
                            date_start, date_end
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (session_key) DO NOTHING
                    """, (
                        session_info.get('session_key'),
                        meeting_info.get('meeting_key'),
                        session_info.get('session_name'),
                        session_info.get('session_type'),
                        session_info.get('date_start'),
                        session_info.get('date_end')
                    ))
                    
                    # Insert drivers for this session - OPTIMIZED FIELDS ONLY
                    if "drivers" in session_data and session_data["drivers"]:
                        for driver in session_data["drivers"]:
                            cursor.execute("""
                                INSERT INTO drivers (
                                    session_key, meeting_key, driver_number,
                                    full_name, team_name
                                ) VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                driver.get('driver_number'),
                                driver.get('full_name'),
                                driver.get('team_name')
                            ))
                    
                    # Insert laps for this session - OPTIMIZED FIELDS ONLY
                    if "laps" in session_data and session_data["laps"]:
                        for lap in session_data["laps"]:
                            cursor.execute("""
                                INSERT INTO laps (
                                    session_key, meeting_key, driver_number, lap_number,
                                    lap_duration, duration_sector_1, duration_sector_2, 
                                    duration_sector_3, is_pit_out_lap
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                lap.get('driver_number'),
                                lap.get('lap_number'),
                                lap.get('lap_duration'),
                                lap.get('duration_sector_1'),
                                lap.get('duration_sector_2'),
                                lap.get('duration_sector_3'),
                                lap.get('is_pit_out_lap')
                            ))
                    
                    # Insert pit stops for this session - OPTIMIZED FIELDS ONLY
                    if "pit" in session_data and session_data["pit"]:
                        for pit in session_data["pit"]:
                            cursor.execute("""
                                INSERT INTO pit_stops (
                                    session_key, meeting_key, driver_number, 
                                    lap_number, pit_duration
                                ) VALUES (%s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                pit.get('driver_number'),
                                pit.get('lap_number'),
                                pit.get('pit_duration')
                            ))
                    
                    # Insert stints for this session - OPTIMIZED FIELDS ONLY
                    if "stints" in session_data and session_data["stints"]:
                        for stint in session_data["stints"]:
                            cursor.execute("""
                                INSERT INTO stints (
                                    session_key, meeting_key, driver_number,
                                    compound, lap_start, lap_end, tyre_age_at_start
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                stint.get('driver_number'),
                                stint.get('compound'),
                                stint.get('lap_start'),
                                stint.get('lap_end'),
                                stint.get('tyre_age_at_start')
                            ))
                    
                    # Insert positions for this session - OPTIMIZED FIELDS ONLY
                    if "position" in session_data and session_data["position"]:
                        for pos in session_data["position"]:
                            cursor.execute("""
                                INSERT INTO positions (
                                    session_key, meeting_key, driver_number, 
                                    position, date
                                ) VALUES (%s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                pos.get('driver_number'),
                                pos.get('position'),
                                pos.get('date')
                            ))
                    
                    # Insert weather for this session - OPTIMIZED FIELDS ONLY
                    if "weather" in session_data and session_data["weather"]:
                        for weather in session_data["weather"]:
                            cursor.execute("""
                                INSERT INTO weather (
                                    session_key, meeting_key, air_temperature,
                                    track_temperature, humidity, rainfall, date
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                weather.get('air_temperature'),
                                weather.get('track_temperature'),
                                weather.get('humidity'),
                                bool(weather.get('rainfall', 0)),  # Convert int to bool
                                weather.get('date')
                            ))
                    
                    if "intervals" in session_data and session_data["intervals"]:
                        for interval in session_data["intervals"]:
                            # Parse the interval values
                            gap_to_leader = self._parse_interval_value(interval.get('gap_to_leader'))
                            interval_val = self._parse_interval_value(interval.get('interval'))
                            
                            cursor.execute("""
                                INSERT INTO intervals (
                                    session_key, meeting_key, driver_number, 
                                    gap_to_leader, interval, date
                                ) VALUES (%s, %s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                interval.get('driver_number'),
                                gap_to_leader,
                                interval_val,
                                interval.get('date')
                            ))
                    
                    if "race_control" in session_data and session_data["race_control"]:
                        for race_control in session_data["race_control"]:
                            cursor.execute("""
                                INSERT INTO race_control (
                                    session_key, meeting_key, driver_number,
                                    category, flag, lap_number, message, 
                                    scope, sector, date
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                session_info.get('session_key'),
                                meeting_info.get('meeting_key'),
                                race_control.get('driver_number'),
                                race_control.get('category'),
                                race_control.get('flag'),
                                race_control.get('lap_number'),
                                race_control.get('message'),
                                race_control.get('scope'),
                                race_control.get('sector'),
                                race_control.get('date')
                            ))
            
            # IMPORTANT: Explicitly commit the transaction
            conn.commit()
            self.logger.info(f"Successfully inserted meeting {meeting_info.get('meeting_key')} data with optimized schema")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error inserting meeting data: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

class DataIngestion:
    """Main data ingestion"""
    
    def __init__(self, data_ingestion_config: DataIngestionConfig, api_client: OpenF1APIClient):
        self.data_ingestion_config = data_ingestion_config
        self.api_client = api_client
        self.logger = logger 
    
    # Add this method to your DataIngestion class (around line 150)
    def _fetch_endpoint_data(self, endpoint: str, session_key: str) -> List[Dict[str, Any]]:
        """Fetch data for a specific endpoint"""
        endpoint_methods = {
            "sessions": lambda: self.api_client.get_sessions(session_key),
            "drivers": lambda: self.api_client.get_drivers(session_key),
            "laps": lambda: self.api_client.get_laps(session_key),
            "pit": lambda: self.api_client.get_pit_stops(session_key),
            "stints": lambda: self.api_client.get_stints(session_key),
            "position": lambda: self.api_client.get_positions(session_key),
            "intervals": lambda: self.api_client.get_intervals(session_key),
            "race_control": lambda: self.api_client.get_race_control(session_key),
            "weather": lambda: self.api_client.get_weather(session_key)
        }
        
        if endpoint in endpoint_methods:
            return endpoint_methods[endpoint]()
        else:
            self.logger.warning(f"Unknown endpoint: {endpoint}")
            return []

    
    def ingest_meeting_data(self, meeting_key: str) -> Dict[str, Any]:
        """Ingest all data for a specific meeting"""
        self.logger.info(f"Starting data ingestion for meeting: {meeting_key}")
        
        # Get meeting info
        meetings = self.api_client.get_meetings()
        meeting_info = next((m for m in meetings if str(m.get('meeting_key')) == meeting_key), None)
        
        if not meeting_info:
            self.logger.error(f"Meeting {meeting_key} not found")
            return {}
        
        # Get sessions for this meeting
        sessions = self.api_client.get_sessions(meeting_key)
        self.logger.info(f"Found {len(sessions)} sessions for meeting {meeting_key}")
        
        meeting_data = {
            "meeting_info": meeting_info,
            "sessions": {},
        }
        
        # Ingest data for each session
        for session in sessions:
            session_name = session.get('session_name', 'Unknown')
            session_key = session.get('session_key')
            
            if session_name not in self.data_ingestion_config.sessions_to_ingest:
                self.logger.info(f"Skipping session: {session_name}")
                continue
            
            self.logger.info(f"Ingesting data for session: {session_name}")
            session_data = self._ingest_session_data(session_key, session_name, meeting_key)
            
            # Add session data to meeting_data
            meeting_data["sessions"][session_name] = session_data
        
        return meeting_data
    

    def _ingest_session_data(self, session_key: str, session_name: str, meeting_key: str) -> Dict[str, Any]:
        """Ingest data for a specific session"""
        session_data = {}
        
        for endpoint in self.data_ingestion_config.endpoints:
            try:
                if endpoint == "meetings":
                    continue
                
                data = self._fetch_endpoint_data(endpoint, session_key)
                session_data[endpoint] = data
                
                # Remove file saving - data will go to database
                self.logger.info(f"Retrieved {len(data)} records from {endpoint} for session {session_key}")
                
            except Exception as e:
                self.logger.error(f"Error ingesting {endpoint} for session {session_key}: {e}")
                session_data[endpoint] = []
        
        return session_data