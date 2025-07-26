import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import psycopg2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.formula_one.logging import logger
from src.formula_one.entity.config_entity import DataTransformationConfig, DatabaseConfig
from src.formula_one.components.data_ingestion import DatabaseIngestion
from src.formula_one.components.base_component import BaseComponent

class DataTransformation(BaseComponent):
    """Handles data transformation for F1 data - Database-First Approach"""
    
    def __init__(self, transformation_config: DataTransformationConfig, db_config: DatabaseConfig):
        super().__init__(transformation_config, db_config)
        
        # Initialize transformers
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
        # Store transformed data
        self.transformed_data = {}
        self.feature_columns = []
    
    def transform_all_data(self) -> Dict[str, pd.DataFrame]:
        """Transform all tables and update database"""
        self.logger.info("Starting comprehensive data transformation (Database-First)")
        
        # Create transformed tables in database
        self._create_transformed_tables()

        self._debug_table_structure('stints')
        
        # Transform and update each table
        for table in self.config.tables_to_transform:
            self.logger.info(f"Transforming table: {table}")
            try:
                df = self._load_table_data(table)
                if df is not None and not df.empty:
                    transformed_df = self._transform_table(df, table)
                    self._update_database_table(table, transformed_df)
                    self.logger.info(f"Successfully transformed and updated {table}")
                else:
                    self.logger.warning(f"Table {table} is empty or could not be loaded")
            except Exception as e:
                self.logger.error(f"Error transforming table {table}: {str(e)}")
                # Continue with other tables instead of stopping
                continue
        
        # Load all transformed data for return
        self.transformed_data = self._load_all_transformed_data()
        
        return self.transformed_data
    
    def _debug_table_structure(self, table_name: str):
        """Debug table structure to understand what columns exist"""
        columns = self.get_table_structure(table_name)
        
        self.logger.info(f"Table structure for {table_name}:")
        for col in columns:
            self.logger.info(f"  {col[0]}: {col[1]} (nullable: {col[2]})")
    
    def _create_transformed_tables(self):
        """Create transformed tables in database"""
        conn = self.db_utils.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # Create transformed tables for each original table
            tables_config = {
                'laps': [
                    "lap_time_std FLOAT",
                    "lap_time_mean FLOAT", 
                    "lap_time_deviation FLOAT",
                    "total_sector_time FLOAT",
                    "sector_consistency FLOAT",
                    "had_incident BOOLEAN",
                    "safety_car_lap BOOLEAN",
                    "is_outlier BOOLEAN",
                ],
                'pit_stops': [
                    "pit_stop_count INTEGER",
                    "pit_stop_timing INTEGER",
                    "normal_pit_stop BOOLEAN",
                    "long_pit_stop BOOLEAN", 
                    "penalty_pit_stop BOOLEAN",
                    "is_outlier BOOLEAN",
                ],
                'stints': [
                    "stint_duration INTEGER",
                    "tire_age_progression INTEGER",
                    "is_outlier BOOLEAN",
                ],
                'positions': [
                    "position_change INTEGER",
                    "position_std FLOAT",
                    "is_leader BOOLEAN",
                    "position_improved BOOLEAN",
                    "position_declined BOOLEAN",
                    "is_outlier BOOLEAN",
                ],
                'intervals': [
                    "is_leader BOOLEAN",
                    "is_lapped BOOLEAN",
                    "is_outlier BOOLEAN"
                ],
                'weather': [
                    "temperature_delta FLOAT",
                    "weather_severity FLOAT",
                    "extreme_weather BOOLEAN",
                ],
                'drivers': [
                    "team_name_encoded INTEGER",
                ],
                'sessions': [
                    "session_type_encoded INTEGER",
                ],
            }
            
            for table_name, new_columns in tables_config.items():
                # Create transformed table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name}_transformed AS 
                    SELECT * FROM {table_name}
                """)
                
                # Add new columns for transformed features
                for column_def in new_columns:
                    try:
                        cursor.execute(f"ALTER TABLE {table_name}_transformed ADD COLUMN IF NOT EXISTS {column_def}")
                    except Exception as e:
                        self.logger.warning(f"Could not add column {column_def} to {table_name}_transformed: {e}")
            
            conn.commit()
            self.logger.info("Created transformed tables in database")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error creating transformed tables: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def _load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Load table data with duplicate handling"""
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, self.db_utils.connect_to_db())
            
            if df.empty:
                self.logger.warning(f"Table {table_name} is empty")
                return None
            
            # Log initial data quality
            self.logger.info(f"Loaded {len(df)} rows from {table_name}")
            
            # Check for duplicates before processing
            if table_name == 'pit_stops':
                duplicates = df.duplicated(subset=['session_key', 'driver_number', 'lap_number']).sum()
            elif table_name == 'laps':
                duplicates = df.duplicated(subset=['session_key', 'driver_number', 'lap_number']).sum()
            elif table_name == 'positions':
                duplicates = df.duplicated(subset=['session_key', 'driver_number', 'date']).sum()
            else:
                duplicates = df.duplicated().sum()
            
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate records in {table_name}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading table {table_name}: {e}")
            return None
    
    def _transform_table(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Transform a specific table"""
        self.logger.info(f"Transforming {table_name} with {len(df)} rows")
        
        try:
            # Step 1: Handle missing values
            df = self._handle_missing_values(df, table_name)
            
            # Step 2: Fix data types
            df = self._fix_data_types(df, table_name)
            
            # Step 3: Handle outliers
            df = self._handle_outliers(df, table_name)
            
            # Step 4: Encode categorical variables
            df = self._encode_categorical_variables(df, table_name)
            
            # Step 5: Create table-specific features
            df = self._create_table_specific_features(df, table_name)
            
            # Debug the final result
            self._debug_table_data(df, table_name)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in _transform_table for {table_name}: {e}")
            # Return the original dataframe with basic transformations
            # Only add is_outlier for tables that have it in schema
            tables_with_outliers = ['laps', 'pit_stops', 'stints', 'positions', 'intervals']
            if table_name in tables_with_outliers:
                df['is_outlier'] = False
                df['outlier_type'] = 'normal'
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Handle missing values based on strategy"""
        self.logger.info(f"Handling missing values in {table_name}")
        
        try:
            if self.config.missing_value_strategy == "impute":
                # Use different strategies for different column types
                for column in df.columns:
                    if column in self.config.numeric_columns and df[column].dtype in ['float64', 'int64']:
                        # Use median for numeric columns
                        try:
                            imputer = SimpleImputer(strategy='median')
                            df[column] = imputer.fit_transform(df[[column]])
                            self.imputers[f"{table_name}_{column}"] = imputer
                        except Exception as e:
                            self.logger.warning(f"Could not impute {column} in {table_name}: {e}")
                            # Fallback to forward fill then backward fill
                            df[column] = df[column].ffill().bfill().fillna(0)
                            
                    elif column in self.config.categorical_columns:
                        # Use most frequent for categorical columns
                        try:
                            imputer = SimpleImputer(strategy='most_frequent')
                            df[column] = imputer.fit_transform(df[[column]])
                            self.imputers[f"{table_name}_{column}"] = imputer
                        except Exception as e:
                            self.logger.warning(f"Could not impute {column} in {table_name}: {e}")
                            # Fallback to 'Unknown'
                            df[column] = df[column].fillna('Unknown')
                            
                    elif column in self.config.datetime_columns:
                        # Forward fill for datetime columns
                        df[column] = df[column].ffill().bfill()

            
            elif self.config.missing_value_strategy == "drop":
                df = df.dropna()
            
            elif self.config.missing_value_strategy == "interpolate":
                # Interpolate numeric columns
                try:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
                except Exception as e:
                    self.logger.warning(f"Could not interpolate numeric columns in {table_name}: {e}")
                    # Fallback to forward fill
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in _handle_missing_values for {table_name}: {e}")
            # Return dataframe with basic missing value handling
            for column in df.columns:
                if df[column].dtype in ['float64', 'int64']:
                    df[column] = df[column].fillna(0)
                elif df[column].dtype == 'object':
                    df[column] = df[column].fillna('Unknown')
            return df
    
    def _fix_data_types(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Fix data types for each column"""
        self.logger.info(f"Fixing data types in {table_name}")
        
        for column in df.columns:
            if column in self.config.numeric_columns:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except:
                    self.logger.warning(f"Could not convert {column} to numeric in {table_name}")
            
            elif column in self.config.datetime_columns:
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                except:
                    self.logger.warning(f"Could not convert {column} to datetime in {table_name}")
            
            elif column in self.config.categorical_columns:
                df[column] = df[column].astype('category')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Handle outliers with F1-specific context"""
        if self.config.outlier_strategy == "ignore":
            return df
        
        self.logger.info(f"Handling outliers in {table_name} with F1 context")

        tables_with_outliers = ['laps', 'pit_stops', 'stints', 'positions', 'intervals']
        
        # Detect F1-specific outliers
        df = self._detect_f1_outliers(df, table_name)
        
        # Handle based on strategy
        if self.config.outlier_strategy == "context_aware":
            # Flag outliers but keep all data
            df = self._flag_outliers_context_aware(df, table_name)
            
        elif self.config.outlier_strategy == "remove_system_errors":
            # Only remove obvious system errors
            df = self._remove_only_system_errors(df, table_name)
            
        elif self.config.outlier_strategy == "cap":
            # Cap outliers to reasonable bounds
            df = self._cap_outliers_f1_context(df, table_name)
        
        return df
    
    def _detect_f1_outliers(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Detect outliers with F1-specific rules"""
        
        if table_name == "laps":
            # Lap time outliers
            df['is_outlier'] = (
                (df['lap_duration'] < 0) | 
                (df['lap_duration'] > 9999) |
                (df['lap_duration'] == 0) |
                (df['lap_duration'] > 300)
            ).astype(bool)
            
            # Categorize outlier types
            df['outlier_type'] = 'normal'
            df.loc[df['lap_duration'] < 0, 'outlier_type'] = 'negative_time'
            df.loc[df['lap_duration'] > 9999, 'outlier_type'] = 'system_error'
            df.loc[(df['lap_duration'] == 0) & (df['lap_duration'] != 0), 'outlier_type'] = 'zero_time'
            df.loc[df['lap_duration'] > 300, 'outlier_type'] = 'incident_lap'
            
        elif table_name == "pit_stops":
            # Pit stop duration outliers
            df['is_outlier'] = (
                (df['pit_duration'] < 0) |
                (df['pit_duration'] > 300) |
                (df['pit_duration'] == 0)
            ).astype(bool)
            
            df['outlier_type'] = 'normal'
            df.loc[df['pit_duration'] < 0, 'outlier_type'] = 'negative_duration'
            df.loc[df['pit_duration'] > 300, 'outlier_type'] = 'penalty_stop'
            df.loc[df['pit_duration'] == 0, 'outlier_type'] = 'zero_duration'
            
        elif table_name == "stints":
            # Stint duration outliers
            df['is_outlier'] = (
                (df['lap_end'] < df['lap_start']) |
                (df['lap_end'] - df['lap_start'] > 100) |
                (df['lap_end'] - df['lap_start'] < 0)
            ).astype(bool)
            
            df['outlier_type'] = 'normal'
            df.loc[df['lap_end'] < df['lap_start'], 'outlier_type'] = 'invalid_range'
            df.loc[df['lap_end'] - df['lap_start'] > 100, 'outlier_type'] = 'extremely_long'
            df.loc[df['lap_end'] - df['lap_start'] < 0, 'outlier_type'] = 'negative_duration'
            
        elif table_name == "positions":
            # Position outliers
            df['is_outlier'] = (
                (df['position'] < 1) |
                (df['position'] > 50) |
                (df['position'] == 0)
            ).astype(bool)
            
            df['outlier_type'] = 'normal'
            df.loc[df['position'] < 1, 'outlier_type'] = 'invalid_position'
            df.loc[df['position'] > 50, 'outlier_type'] = 'too_many_cars'
            df.loc[df['position'] == 0, 'outlier_type'] = 'zero_position'
            
        elif table_name == "intervals":
            # Interval outliers
            df['is_outlier'] = (
                (df['gap_to_leader'] < -999) |
                (df['interval'] < -999) |
                (df['gap_to_leader'] > 9999) |
                (df['interval'] > 9999)
            ).astype(bool)
            
            df['outlier_type'] = 'normal'
            df.loc[df['gap_to_leader'] < -999, 'outlier_type'] = 'lapped_car'
            df.loc[df['interval'] < -999, 'outlier_type'] = 'lapped_car'
            df.loc[df['gap_to_leader'] > 9999, 'outlier_type'] = 'system_error'
            df.loc[df['interval'] > 9999, 'outlier_type'] = 'system_error'
        
        return df
    
    def _flag_outliers_context_aware(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Flag outliers but keep all data for context-aware analysis"""
        # Outliers are already flagged in _detect_f1_outliers
        # This method just ensures we keep all data
        return df
    
    def _remove_only_system_errors(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Remove only obvious system errors, keep legitimate outliers"""
        if table_name == "laps":
            # Remove only system errors, keep incident laps
            df = df[df['outlier_type'] != 'system_error']
        elif table_name == "pit_stops":
            # Remove only zero duration, keep penalties
            df = df[df['outlier_type'] != 'zero_duration']
        elif table_name == "stints":
            # Remove only invalid ranges
            df = df[df['outlier_type'] != 'invalid_range']
        elif table_name == "positions":
            # Remove only zero positions
            df = df[df['outlier_type'] != 'zero_position']
        elif table_name == "intervals":
            # Remove only system errors, keep lapped cars
            df = df[df['outlier_type'] != 'system_error']
        
        return df
    
    def _cap_outliers_f1_context(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Cap outliers to reasonable bounds for F1 context"""
        if table_name == "laps":
            # Cap lap times to reasonable bounds
            df['lap_duration'] = df['lap_duration'].clip(lower=0, upper=300)
        elif table_name == "pit_stops":
            # Cap pit stop durations
            df['pit_duration'] = df['pit_duration'].clip(lower=0.5, upper=300)
        elif table_name == "stints":
            # Cap stint durations
            df['stint_duration'] = (df['lap_end'] - df['lap_start']).clip(lower=0, upper=100)
        elif table_name == "positions":
            # Cap positions
            df['position'] = df['position'].clip(lower=1, upper=50)
        elif table_name == "intervals":
            # Cap intervals
            df['gap_to_leader'] = df['gap_to_leader'].clip(lower=-999, upper=9999)
            df['interval'] = df['interval'].clip(lower=-999, upper=9999)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder"""
        self.logger.info(f"Encoding categorical variables in {table_name}")
        
        # Define which columns to encode for each table
        encoding_config = {
            'sessions': ['session_type'],
            'drivers': ['team_name'],
        }
        
        if table_name in encoding_config:
            for column in encoding_config[table_name]:
                if column in df.columns:
                    # Handle missing values first - use a different approach for categorical
                    if df[column].dtype.name == 'category':
                        # If it's already categorical, add 'Unknown' to categories first
                        current_categories = df[column].cat.categories.tolist()
                        if 'Unknown' not in current_categories:
                            df[column] = df[column].cat.add_categories(['Unknown'])
                        df[column] = df[column].fillna('Unknown')
                    else:
                        # Convert to string and fill missing values
                        df[column] = df[column].astype(str).fillna('Unknown')
                    
                    # Create label encoder
                    le = LabelEncoder()
                    try:
                        # Fit and transform
                        encoded_values = le.fit_transform(df[column].astype(str))
                        df[f"{column}_encoded"] = encoded_values
                        self.label_encoders[f"{table_name}_{column}"] = le
                        
                        self.logger.info(f"Encoded {column} in {table_name} with {len(le.classes_)} unique values")
                    except Exception as e:
                        self.logger.warning(f"Could not encode {column} in {table_name}: {e}")
                        df[f"{column}_encoded"] = 0  # Default value
                else:
                    self.logger.warning(f"Column {column} not found in {table_name}")
                    df[f"{column}_encoded"] = 0  # Default value
        
        return df
    
    def _create_table_specific_features(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Create table-specific features"""
        try:
            if table_name == "laps":
                df = self._create_lap_features(df)
            elif table_name == "stints":
                df = self._create_stint_features(df)
            elif table_name == "pit_stops":
                df = self._create_pit_stop_features(df)
            elif table_name == "positions":
                df = self._create_position_features(df)
            elif table_name == "intervals":
                df = self._create_interval_features(df)
            elif table_name == "weather":
                df = self._create_weather_features(df)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error in _create_table_specific_features for {table_name}: {e}")
            return df
    
    def _create_lap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lap-specific features"""
        # Lap time consistency
        df['lap_time_std'] = df.groupby('driver_number')['lap_duration'].transform('std')
        df['lap_time_mean'] = df.groupby('driver_number')['lap_duration'].transform('mean')
        df['lap_time_deviation'] = df['lap_duration'] - df['lap_time_mean']
        
        # Sector analysis
        df['total_sector_time'] = df['duration_sector_1'] + df['duration_sector_2'] + df['duration_sector_3']
        df['sector_consistency'] = df[['duration_sector_1', 'duration_sector_2', 'duration_sector_3']].std(axis=1)
        
        # Boolean features for laps
        df['had_incident'] = ((df['lap_duration'] > 120) & (df['lap_duration'] <= 300)).astype(bool)
        df['safety_car_lap'] = ((df['lap_duration'] > 90) & (df['lap_duration'] <= 120)).astype(bool)

        return df
    
    def _create_stint_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create stint-specific features"""
        try:
            # Ensure we have the required columns
            required_cols = ['lap_start', 'lap_end', 'tyre_age_at_start', 'session_key', 'driver_number']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns in stints: {missing_cols}")
                # Add default values for missing columns
                for col in missing_cols:
                    df[col] = 0
            
            # Stint duration - handle missing values and invalid data
            df['stint_duration'] = df['lap_end'] - df['lap_start']
            # Handle cases where lap_end < lap_start (invalid data)
            df['stint_duration'] = df['stint_duration'].clip(lower=0, upper=100)
            df['stint_duration'] = df['stint_duration'].fillna(0)
            
            # Tire age progression - handle missing values
            try:
                df['tire_age_progression'] = df.groupby(['session_key', 'driver_number'])['tyre_age_at_start'].diff()
                df['tire_age_progression'] = df['tire_age_progression'].fillna(0).clip(lower=0, upper=50)
            except Exception as e:
                self.logger.warning(f"Could not calculate tire_age_progression: {e}")
                df['tire_age_progression'] = 0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in _create_stint_features: {e}")
            # Return dataframe with default values
            df['stint_duration'] = 0
            df['tire_age_progression'] = 0
            return df
    
    def _create_pit_stop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pit stop-specific features with ALL features included"""

        df['pit_stop_count'] = df.groupby(['session_key', 'driver_number']).cumcount() + 1
        df['pit_stop_count'] = df['pit_stop_count'].clip(upper=20)  # Cap at 20 pit stops (more reasonable)
        
        df['pit_stop_timing'] = df.groupby(['session_key', 'driver_number'])['lap_number'].diff()
        df['pit_stop_timing'] = df['pit_stop_timing'].clip(lower=0, upper=50)  # Cap at 50 laps

        df['normal_pit_stop'] = (df['pit_duration'] >= 0.5) & (df['pit_duration'] <= 5)
        df['long_pit_stop'] = (df['pit_duration'] > 5) & (df['pit_duration'] <= 60)
        df['penalty_pit_stop'] = (df['pit_duration'] > 60) & (df['pit_duration'] <= 300)
        
        return df

    def _create_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features with ALL features included"""

        df['position_change'] = df.groupby(['session_key', 'driver_number'])['position'].diff()
        df['position_change'] = df['position_change'].clip(lower=-20, upper=20) 
        
        df['position_std'] = df.groupby(['session_key', 'driver_number'])['position'].transform('std')
        df['position_std'] = df['position_std'].clip(upper=20) 
        
        df['is_leader'] = (df['position'] == 1).astype(bool) 

        # Position improvement/decline
        df['position_improved'] = (df['position_change'] < 0).astype(bool)
        df['position_declined'] = (df['position_change'] > 0).astype(bool)
        
        return df

    def _create_interval_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interval-specific features with ALL features included"""
        # Boolean features (these should be created here, not in outlier handling)
        df['is_leader'] = df['gap_to_leader'] == 0
        df['is_lapped'] = (df['gap_to_leader'] == -999) | (df['interval'] == -999)
        
        return df
    
    def _create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-specific features"""
        # Weather impact features
        df['temperature_delta'] = df['track_temperature'] - df['air_temperature']
        df['weather_severity'] = df['humidity'] * df['rainfall'].astype(int)
        
        # Boolean features for weather
        df['extreme_weather'] = (
            ((df['air_temperature'] < -20) | (df['air_temperature'] > 50)) |
            ((df['track_temperature'] < -10) | (df['track_temperature'] > 80)) |
            (df['humidity'] > 90)
        ).astype(bool)
        
        return df
    
    def _update_database_table(self, table_name: str, df: pd.DataFrame):
        """Update transformed table in database"""
        conn = self.db_utils.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # First, get the actual columns that exist in the database table
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}_transformed'
                ORDER BY ordinal_position
            """)
            db_columns = [row[0] for row in cursor.fetchall()]
            
            # Filter DataFrame to only include columns that exist in the database
            df_clean = df[db_columns].copy()
            
            # Convert numeric columns to appropriate types
            for col in df_clean.columns:
                if col in ['pit_duration', 'lap_duration', 'duration_sector_1', 'duration_sector_2', 'duration_sector_3']:
                    # Time-related columns should be floats
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0).astype('float32')
                elif col in ['pit_stop_count', 'pit_stop_timing', 'stint_duration', 'tire_age_progression', 
                            'position_change', 'lap_number', 'driver_number', 'session_key', 'meeting_key']:
                    # Count/timing columns should be integers
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('int32')
                elif 'std' in col or 'mean' in col or 'deviation' in col or 'consistency' in col:
                    # Statistical columns should be floats
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0).astype('float32')
                elif col in ['is_outlier', 'had_incident', 'safety_car_lap', 
                            'normal_pit_stop', 'long_pit_stop', 'penalty_pit_stop', 'is_leader', 'is_retired', 
                            'is_lapped', 'extreme_weather']:
                    # Boolean columns
                    df_clean[col] = df_clean[col].fillna(False).astype('bool')
                elif col.endswith('_encoded'):
                    # Encoded columns should be integers
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('int32')
                elif col == 'outlier_type':
                    # String column
                    df_clean[col] = df_clean[col].fillna('normal')
            
            df_clean = self._remove_duplicates(df_clean, table_name)
            
            # Clear existing data
            cursor.execute(f"DELETE FROM {table_name}_transformed")
            
            # Insert transformed data
            for _, row in df_clean.iterrows():
                placeholders = ', '.join(['%s'] * len(row))
                columns = ', '.join(row.index)
                query = f"INSERT INTO {table_name}_transformed ({columns}) VALUES ({placeholders})"
                cursor.execute(query, tuple(row.values))
            
            conn.commit()
            self.logger.info(f"Updated {table_name}_transformed with {len(df)} rows")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating {table_name}_transformed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def _remove_duplicates(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Remove duplicates based on table-specific logic"""
        
        if table_name == 'pit_stops':
            # Remove duplicates based on session_key, driver_number, lap_number
            df_clean = df.drop_duplicates(
                subset=['session_key', 'driver_number', 'lap_number'], 
                keep='first'
            )
            self.logger.info(f"Removed {len(df) - len(df_clean)} duplicate pit stop records")
            
        elif table_name == 'laps':
            # Remove duplicates based on session_key, driver_number, lap_number
            df_clean = df.drop_duplicates(
                subset=['session_key', 'driver_number', 'lap_number'], 
                keep='first'
            )
            self.logger.info(f"Removed {len(df) - len(df_clean)} duplicate lap records")
            
        elif table_name == 'positions':
            # Remove duplicates based on session_key, driver_number, date
            df_clean = df.drop_duplicates(
                subset=['session_key', 'driver_number', 'date'], 
                keep='first'
            )
            self.logger.info(f"Removed {len(df) - len(df_clean)} duplicate position records")
            
        elif table_name == 'stints':
            # Remove duplicates based on session_key, driver_number, lap_start, lap_end
            df_clean = df.drop_duplicates(
                subset=['session_key', 'driver_number', 'lap_start', 'lap_end'], 
                keep='first'
            )
            self.logger.info(f"Removed {len(df) - len(df_clean)} duplicate stint records")
            
        else:
            # For other tables, remove exact duplicates
            df_clean = df.drop_duplicates()
            self.logger.info(f"Removed {len(df) - len(df_clean)} duplicate records from {table_name}")
        
        return df_clean
    
    def _load_all_transformed_data(self) -> Dict[str, pd.DataFrame]:
        """Load all transformed data from database"""
        transformed_data = {}
        
        for table in self.config.tables_to_transform:
            df = self._load_table_data(f"{table}_transformed")
            if df is not None:
                transformed_data[table] = df
        
        return transformed_data
    
    def _debug_table_data(self, df: pd.DataFrame, table_name: str):
        """Debug table data after transformation"""
        self.logger.info(f"Final transformed data for {table_name}:")
        self.logger.info(f"  Shape: {df.shape}")
        self.logger.info(f"  Columns: {list(df.columns)}")
        
        # Log encoded columns if any
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        if encoded_cols:
            self.logger.info(f"  Encoded columns: {encoded_cols}")
            for col in encoded_cols:
                value_counts = df[col].value_counts()
                self.logger.info(f"    {col}: {dict(value_counts)}")
        
        # Log sample data
        self.logger.info(f"  Sample data (first 3 rows):")
        self.logger.info(f"    {df.head(3).to_dict('records')}") 