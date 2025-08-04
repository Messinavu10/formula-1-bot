import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.formula_one.logging import logger
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

@dataclass
class ValidationResult:
    """Result of validation checks"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class QueryValidator:
    """Comprehensive query validation for F1 data analysis"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.logger = logging.getLogger(__name__)
        
        # Valid driver names (case-insensitive)
        self.valid_drivers = {
            'max verstappen',
            'liam lawson',
            'charles leclerc',
            'lewis hamilton',
            'george russell',
            'kimi antonelli',
            'lando norris',
            'oscar piastri',
            'fernando alonso',
            'lance stroll',
            'pierre gasly',
            'franco colapinto',
            'oliver bearman',
            'nico hulkenberg',
            'gabriel bortoleto',
            'yuki tsunoda',
            'esteban ocon',
            'alex albon',
            'isack hadjar'
        }
        
        # Valid team names (case-insensitive)
        self.valid_teams = {
            'red bull', 'red bull racing', 'mercedes', 'ferrari', 'mclaren',
            'aston martin', 'alpine', 'haas', 'racing bulls', 'sauber', 'williams'
        }
        
        # Valid races (case-insensitive)
        self.valid_races = {
            'bahrain grand prix', 'saudi arabian grand prix', 'australian grand prix',
            'japanese grand prix', 'chinese grand prix', 'miami grand prix',
            'emilia romagna grand prix', 'monaco grand prix', 'canadian grand prix',
            'spanish grand prix', 'austrian grand prix', 'british grand prix',
            'hungarian grand prix', 'belgian grand prix', 'dutch grand prix',
            'italian grand prix', 'azerbaijan grand prix', 'singapore grand prix',
            'united states grand prix', 'mexican grand prix', 'brazilian grand prix',
            'las vegas grand prix', 'qatar grand prix', 'abu dhabi grand prix'
        }
        
        # Valid session types
        self.valid_sessions = {
            'practice', 'qualifying', 'sprint', 'race', 'sprint qualifying',
            'sprint shootout', 'practice 1', 'practice 2', 'practice 3',
            'fp1', 'fp2', 'fp3', 'q1', 'q2', 'q3', 'sq1', 'sq2', 'sq3'
        }
        
        # Maximum limits
        self.max_drivers = 5
        self.max_teams = 5
        self.max_query_length = 500
        self.min_query_length = 3

    def validate_query(self, user_query: str, query_analysis: Dict[str, Any]) -> ValidationResult:
        """Comprehensive query validation"""
        errors = []
        warnings = []
        suggestions = []
        
        # Handle case where query_analysis is None or incomplete
        if not query_analysis:
            errors.append("Query analysis failed. Please try rephrasing your query.")
            suggestions.append("Try: 'lap time progression for Max Verstappen at Monaco GP'")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Ensure query_analysis is a dictionary
        if not isinstance(query_analysis, dict):
            errors.append("Query analysis format error. Please try rephrasing your query.")
            suggestions.append("Try: 'lap time progression for Max Verstappen at Monaco GP'")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Basic query validation
        basic_result = self._validate_basic_query(user_query)
        errors.extend(basic_result.errors)
        warnings.extend(basic_result.warnings)
        suggestions.extend(basic_result.suggestions)
        
        # Entity validation
        entity_result = self._validate_entities(query_analysis)
        errors.extend(entity_result.errors)
        warnings.extend(entity_result.warnings)
        suggestions.extend(entity_result.suggestions)
        
        # Session validation
        session_result = self._validate_session(query_analysis)
        errors.extend(session_result.errors)
        warnings.extend(session_result.warnings)
        suggestions.extend(session_result.suggestions)
        
        # Data availability validation
        data_result = self._validate_data_availability(query_analysis)
        errors.extend(data_result.errors)
        warnings.extend(data_result.warnings)
        suggestions.extend(data_result.suggestions)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

    def _validate_basic_query(self, user_query: str) -> ValidationResult:
        """Validate basic query characteristics"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for empty or minimal queries
        if not user_query or not user_query.strip():
            errors.append("Query cannot be empty. Please provide a valid F1 analysis request.")
            suggestions.append("Try: 'lap time progression for Max Verstappen at Monaco GP'")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Check query length
        if len(user_query.strip()) < self.min_query_length:
            errors.append(f"Query too short (minimum {self.min_query_length} characters).")
            suggestions.append("Try: 'lap time progression for Max Verstappen at Monaco GP'")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Allow follow-up queries like "summary", "chart", "visualize", etc.
        follow_up_keywords = ['summary', 'chart', 'visualize', 'graph', 'plot', 'show', 'display', 'yes', 'no']
        query_lower = user_query.lower().strip()
        if len(query_lower.split()) == 1 and query_lower in follow_up_keywords:
            # This is a valid follow-up query, don't flag it as too short
            return ValidationResult(True, errors, warnings, suggestions)
        
        if len(user_query) > self.max_query_length:
            errors.append(f"Query too long (maximum {self.max_query_length} characters).")
            suggestions.append("Please provide a more concise query.")
            return ValidationResult(False, errors, warnings, suggestions)
        
        # Check for special characters that might cause issues
        special_chars = re.findall(r'[@#$%^&*()+=<>{}[\]|\\:;"\'`~]', user_query)
        if special_chars:
            warnings.append(f"Special characters detected: {special_chars}. These may cause issues.")
            suggestions.append("Consider removing special characters for better results.")
        
        # Check for single character queries
        if len(user_query.strip()) == 1:
            errors.append("Single character queries are not supported.")
            suggestions.append("Try: 'lap time progression for Max Verstappen at Monaco GP'")
            return ValidationResult(False, errors, warnings, suggestions)
        
        return ValidationResult(True, errors, warnings, suggestions)

    def _validate_entities(self, query_analysis: Dict[str, Any]) -> ValidationResult:
        """Validate drivers, teams, and races"""
        errors = []
        warnings = []
        suggestions = []
        
        drivers = query_analysis.get('drivers', [])
        teams = query_analysis.get('teams', [])
        meeting_info = query_analysis.get('meeting_info', {})
        
        # Validate drivers
        invalid_drivers = []
        for driver in drivers:
            if driver and isinstance(driver, str) and driver.lower() not in self.valid_drivers:
                invalid_drivers.append(driver)
        
        if invalid_drivers:
            errors.append(f"Invalid driver name(s): {', '.join(invalid_drivers)}")
            suggestions.append("Valid drivers include: Max Verstappen, Lewis Hamilton, Charles Leclerc, etc.")
        
        # Validate teams
        invalid_teams = []
        for team in teams:
            if team and isinstance(team, str) and team.lower() not in self.valid_teams:
                invalid_teams.append(team)
        
        if invalid_teams:
            errors.append(f"Invalid team name(s): {', '.join(invalid_teams)}")
            suggestions.append("Valid teams include: Red Bull, Mercedes, Ferrari, McLaren, etc.")
        
        # Check driver/team limits
        if len(drivers) > self.max_drivers:
            errors.append(f"Too many drivers specified ({len(drivers)}). Maximum is {self.max_drivers}.")
            suggestions.append("Please select fewer drivers for analysis.")
        
        if len(teams) > self.max_teams:
            errors.append(f"Too many teams specified ({len(teams)}). Maximum is {self.max_teams}.")
            suggestions.append("Please select fewer teams for analysis.")
        
        # Validate race
        race_name = meeting_info.get('name', '')
        if race_name and isinstance(race_name, str):
            race_name = race_name.lower()
            if race_name not in self.valid_races:
                errors.append(f"Invalid race name: {meeting_info.get('name')}")
                suggestions.append("Valid races include: Monaco Grand Prix, British Grand Prix, etc.")
        
        # Check for mixed drivers and teams
        if drivers and teams:
            warnings.append("Mixed drivers and teams detected. This may cause confusion.")
            suggestions.append("Consider focusing on either drivers or teams, not both.")
        
        return ValidationResult(True, errors, warnings, suggestions)

    def _validate_session(self, query_analysis: Dict[str, Any]) -> ValidationResult:
        """Validate session types"""
        errors = []
        warnings = []
        suggestions = []
        
        session_type = query_analysis.get('session_type', '')
        if session_type and hasattr(session_type, 'lower'):
            session_type = session_type.lower()
        
        # Check for invalid session types
        if session_type and session_type not in self.valid_sessions:
            errors.append(f"Invalid session type: {session_type}")
            suggestions.append("Valid sessions include: Race, Practice, Qualifying, Sprint")
        
        # Check for sessions that typically have no data
        no_data_sessions = ['practice', 'fp1', 'fp2', 'fp3']
        if session_type in no_data_sessions:
            warnings.append(f"Session type '{session_type}' typically has limited data available.")
            suggestions.append("Consider using 'Race' session for more comprehensive analysis.")
        
        return ValidationResult(True, errors, warnings, suggestions)

    def _validate_data_availability(self, query_analysis: Dict[str, Any]) -> ValidationResult:
        """Validate data availability for the query"""
        errors = []
        warnings = []
        suggestions = []
        
        # This would typically involve database queries to check actual data availability
        # For now, we'll add basic checks
        
        meeting_info = query_analysis.get('meeting_info', {})
        session_type = query_analysis.get('session_type', '')
        
        # Check for testing sessions (typically no data)
        if 'testing' in meeting_info.get('name', '').lower():
            warnings.append("Testing sessions typically have limited data available.")
            suggestions.append("Consider using a regular race weekend for analysis.")
        
        # Check for practice sessions with specific queries
        if session_type in ['practice', 'fp1', 'fp2', 'fp3']:
            query_type = query_analysis.get('query_type', '')
            if query_type in ['pit_strategy', 'tire_strategy']:
                warnings.append("Pit stop and tire strategy data may be limited in practice sessions.")
                suggestions.append("Consider using 'Race' session for strategy analysis.")
        
        return ValidationResult(True, errors, warnings, suggestions)

    def get_helpful_suggestions(self, query_type: str) -> List[str]:
        """Get helpful suggestions based on query type"""
        suggestions = {
            'driver_performance': [
                "• 'driver performance for Max Verstappen at Monaco GP'",
                "• 'how did Lewis Hamilton perform at British GP'",
                "• 'Charles Leclerc performance at Silverstone'"
            ],
            'team_performance': [
                "• 'team performance for Red Bull at Bahrain GP'",
                "• 'how did Ferrari perform at Silverstone'",
                "• 'Mercedes team analysis at Monaco GP'"
            ],
            'lap_time_progression': [
                "• 'lap time progression for Max Verstappen at Monaco GP'",
                "• 'lap times for Lewis Hamilton at British GP'",
                "• 'lap time analysis for Charles Leclerc at Silverstone'"
            ],
            'sector_analysis': [
                "• 'sector analysis for Charles Leclerc at Monaco GP'",
                "• 'sector performance for Red Bull at Bahrain'",
                "• 'sector times for Lando Norris at Silverstone'"
            ],
            'position_analysis': [
                "• 'position progression for Lando Norris at Silverstone'",
                "• 'position changes for Ferrari at Monaco GP'",
                "• 'position analysis for Max Verstappen at British GP'"
            ],
            'pit_strategy': [
                "• 'pit stop analysis for Max Verstappen at Monaco GP'",
                "• 'pit strategy for Red Bull at British GP'",
                "• 'pit stops for Charles Leclerc at Silverstone'"
            ],
            'tire_strategy': [
                "• 'tire strategy for Lewis Hamilton at Bahrain GP'",
                "• 'tire analysis for Ferrari at Monaco GP'",
                "• 'tire compounds for Lando Norris at Silverstone'"
            ],
            'comparison': [
                "• 'compare Max Verstappen vs Lewis Hamilton at Monaco GP'",
                "• 'Red Bull vs Ferrari performance at Silverstone'",
                "• 'Charles Leclerc vs Lando Norris at British GP'"
            ],
            'visualization': [
                "• 'show me lap time progression for Max Verstappen at Monaco GP'",
                "• 'create a chart for Red Bull performance at Silverstone'",
                "• 'visualize sector analysis for Charles Leclerc at British GP'"
            ]
        }
        
        return suggestions.get(query_type, suggestions.get('general', [
            "• 'lap time progression for Max Verstappen at Monaco GP'",
            "• 'team performance for Red Bull at Bahrain GP'",
            "• 'sector analysis for Charles Leclerc at Silverstone'"
        ]))



    def format_error_response(self, validation_result: ValidationResult, query_type: str = None) -> str:
        """Format a user-friendly error response"""
        if validation_result.is_valid:
            return ""
        
        # Check for specific validation errors and provide direct responses
        for error in validation_result.errors:
            if "Query too short" in error or "Single character queries" in error:
                suggestions = self.get_helpful_suggestions(query_type) if query_type else self.get_helpful_suggestions("general")
                return f"Your query is too short. Please provide a complete query like:\n" + "\n".join(suggestions[:3])
            
            if "Query cannot be empty" in error:
                suggestions = self.get_helpful_suggestions(query_type) if query_type else self.get_helpful_suggestions("general")
                return f"Please provide a query. Try something like:\n" + "\n".join(suggestions[:3])
            
            if "Query too long" in error:
                suggestions = self.get_helpful_suggestions(query_type) if query_type else self.get_helpful_suggestions("general")
                return f"Your query is too long and I can't process that much data at once. Please try a more specific query like:\n" + "\n".join(suggestions[:3])
            
            if "Invalid driver name(s):" in error:
                invalid_drivers = error.replace("Invalid driver name(s):", "").strip().split(", ")
                # Check if it might be a spelling mistake
                if len(invalid_drivers) == 1:
                    driver = invalid_drivers[0]
                    # Suggest similar names
                    suggestions = self._suggest_similar_driver(driver)
                    if suggestions:
                        return f"I don't recognize '{driver}'. Did you mean one of these drivers?\n{suggestions}\n\nValid drivers include: Max Verstappen, Lewis Hamilton, Charles Leclerc, Lando Norris, etc."
                    else:
                        return f"I don't recognize '{driver}'. Valid drivers include: Max Verstappen, Lewis Hamilton, Charles Leclerc, Lando Norris, etc."
                else:
                    return f"I don't recognize the driver name(s): {', '.join(invalid_drivers)}. Valid drivers include: Max Verstappen, Lewis Hamilton, Charles Leclerc, Lando Norris, etc."
            
            if "Invalid team name(s):" in error:
                invalid_teams = error.replace("Invalid team name(s):", "").strip().split(", ")
                # Check if it might be a spelling mistake
                if len(invalid_teams) == 1:
                    team = invalid_teams[0]
                    # Suggest similar names
                    suggestions = self._suggest_similar_team(team)
                    if suggestions:
                        return f"I don't recognize '{team}'. Did you mean one of these teams?\n{suggestions}\n\nValid teams include: Red Bull, Mercedes, Ferrari, McLaren, etc."
                    else:
                        return f"I don't recognize '{team}'. Valid teams include: Red Bull, Mercedes, Ferrari, McLaren, etc."
                else:
                    return f"I don't recognize the team name(s): {', '.join(invalid_teams)}. Valid teams include: Red Bull, Mercedes, Ferrari, McLaren, etc."
            
            if "Invalid race name:" in error:
                invalid_race = error.replace("Invalid race name:", "").strip()
                # Suggest similar race names
                suggestions = self._suggest_similar_race(invalid_race)
                if suggestions:
                    return f"I couldn't find '{invalid_race}'. Did you mean one of these races?\n{suggestions}\n\nValid races include: Monaco Grand Prix, British Grand Prix, Bahrain Grand Prix, etc."
                else:
                    return f"I couldn't find '{invalid_race}' in our database. Valid races include: Monaco Grand Prix, British Grand Prix, Bahrain Grand Prix, etc."
        
        # Generic error for other issues
        suggestions = self.get_helpful_suggestions(query_type) if query_type else self.get_helpful_suggestions("general")
        return f"I couldn't understand your query. Please try something like:\n" + "\n".join(suggestions[:3])

    def _suggest_similar_driver(self, driver_name: str) -> str:
        """Suggest similar driver names for spelling mistakes"""
        driver_lower = driver_name.lower()
        suggestions = []
        
        # Common misspellings and similar names
        similar_drivers = {
            "max": ["Max Verstappen"],
            "verstappen": ["Max Verstappen"],
            "lewis": ["Lewis Hamilton"],
            "hamilton": ["Lewis Hamilton"],
            "charles": ["Charles Leclerc"],
            "leclerc": ["Charles Leclerc"],
            "lando": ["Lando Norris"],
            "norris": ["Lando Norris"],
            "oscar": ["Oscar Piastri"],
            "piastri": ["Oscar Piastri"],
            "george": ["George Russell"],
            "russell": ["George Russell"],
            "fernando": ["Fernando Alonso"],
            "alonso": ["Fernando Alonso"],
            "carlos": ["Carlos Sainz"],
            "sainz": ["Carlos Sainz"],
            "pierre": ["Pierre Gasly"],
            "gasly": ["Pierre Gasly"],
            "yuki": ["Yuki Tsunoda"],
            "tsunoda": ["Yuki Tsunoda"],
            "alex": ["Alex Albon"],
            "albon": ["Alex Albon"],
            "lance": ["Lance Stroll"],
            "stroll": ["Lance Stroll"],
            "esteban": ["Esteban Ocon"],
            "ocon": ["Esteban Ocon"],
            "nico": ["Nico Hulkenberg"],
            "hulkenberg": ["Nico Hulkenberg"],
            "hulk": ["Nico Hulkenberg"],
            "liam": ["Liam Lawson"],
            "lawson": ["Liam Lawson"],
            "oliver": ["Oliver Bearman"],
            "bearman": ["Oliver Bearman"],
            "gabriel": ["Gabriel Bortoleto"],
            "bortoleto": ["Gabriel Bortoleto"],
            "isack": ["Isack Hadjar"],
            "hadjar": ["Isack Hadjar"],
            "jack": ["Jack Doohan"],
            "doohan": ["Jack Doohan"],
            "franco": ["Franco Colapinto"],
            "colapinto": ["Franco Colapinto"],
            "andrea": ["Andrea Kimi Antonelli"],
            "antonelli": ["Andrea Kimi Antonelli"],
            "kimi": ["Andrea Kimi Antonelli"]
        }
        
        for key, drivers in similar_drivers.items():
            if key in driver_lower or driver_lower in key:
                suggestions.extend(drivers)
        
        if suggestions:
            return "• " + "\n• ".join(set(suggestions))
        return ""

    def _suggest_similar_team(self, team_name: str) -> str:
        """Suggest similar team names for spelling mistakes"""
        team_lower = team_name.lower()
        suggestions = []
        
        # Common misspellings and similar names
        similar_teams = {
            "red bull": ["Red Bull Racing"],
            "redbull": ["Red Bull Racing"],
            "red bull racing": ["Red Bull Racing"],
            "mercedes": ["Mercedes"],
            "ferrari": ["Ferrari"],
            "mclaren": ["McLaren"],
            "aston martin": ["Aston Martin"],
            "aston": ["Aston Martin"],
            "alpine": ["Alpine"],
            "haas": ["Haas"],
            "williams": ["Williams"],
            "sauber": ["Sauber"],
            "racing bulls": ["Racing Bulls"],
            "bulls": ["Racing Bulls"],
            "alpha tauri": ["AlphaTauri"],
            "alphatauri": ["AlphaTauri"],
            "alfa romeo": ["Alfa Romeo"],
            "alfa": ["Alfa Romeo"]
        }
        
        for key, teams in similar_teams.items():
            if key in team_lower or team_lower in key:
                suggestions.extend(teams)
        
        if suggestions:
            return "• " + "\n• ".join(set(suggestions))
        return ""

    def _suggest_similar_race(self, race_name: str) -> str:
        """Suggest similar race names for spelling mistakes"""
        race_lower = race_name.lower()
        suggestions = []
        
        # Common misspellings and similar names
        similar_races = {
            "monaco": ["Monaco Grand Prix"],
            "monaco gp": ["Monaco Grand Prix"],
            "monaco grand prix": ["Monaco Grand Prix"],
            "british": ["British Grand Prix"],
            "silverstone": ["British Grand Prix"],
            "british gp": ["British Grand Prix"],
            "british grand prix": ["British Grand Prix"],
            "bahrain": ["Bahrain Grand Prix"],
            "bahrain gp": ["Bahrain Grand Prix"],
            "bahrain grand prix": ["Bahrain Grand Prix"],
            "australian": ["Australian Grand Prix"],
            "australia": ["Australian Grand Prix"],
            "australian gp": ["Australian Grand Prix"],
            "australian grand prix": ["Australian Grand Prix"],
            "saudi": ["Saudi Arabian Grand Prix"],
            "saudi arabian": ["Saudi Arabian Grand Prix"],
            "saudi gp": ["Saudi Arabian Grand Prix"],
            "saudi arabian gp": ["Saudi Arabian Grand Prix"],
            "saudi arabian grand prix": ["Saudi Arabian Grand Prix"],
            "miami": ["Miami Grand Prix"],
            "miami gp": ["Miami Grand Prix"],
            "miami grand prix": ["Miami Grand Prix"],
            "emilia romagna": ["Emilia‑Romagna Grand Prix"],
            "imola": ["Emilia‑Romagna Grand Prix"],
            "emilia romagna gp": ["Emilia‑Romagna Grand Prix"],
            "emilia‑romagna grand prix": ["Emilia‑Romagna Grand Prix"],
            "spanish": ["Spanish Grand Prix"],
            "spain": ["Spanish Grand Prix"],
            "spanish gp": ["Spanish Grand Prix"],
            "spanish grand prix": ["Spanish Grand Prix"],
            "canadian": ["Canadian Grand Prix"],
            "canada": ["Canadian Grand Prix"],
            "canadian gp": ["Canadian Grand Prix"],
            "canadian grand prix": ["Canadian Grand Prix"],
            "austrian": ["Austrian Grand Prix"],
            "austria": ["Austrian Grand Prix"],
            "austrian gp": ["Austrian Grand Prix"],
            "austrian grand prix": ["Austrian Grand Prix"],
            "belgian": ["Belgian Grand Prix"],
            "spa": ["Belgian Grand Prix"],
            "belgian gp": ["Belgian Grand Prix"],
            "belgian grand prix": ["Belgian Grand Prix"]
        }
        
        for key, races in similar_races.items():
            if key in race_lower or race_lower in key:
                suggestions.extend(races)
        
        if suggestions:
            return "• " + "\n• ".join(set(suggestions))
        return ""