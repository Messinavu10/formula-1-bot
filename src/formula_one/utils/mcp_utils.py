from typing import Dict, Any
from src.formula_one.logging import logger

def format_lap_time(seconds: float) -> str:
    """Convert seconds to MM:SS.mmm format"""
    if seconds is None:
        return "N/A"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:06.3f}"

def log_tool_execution(tool_name: str, params: dict, duration: float, success: bool):
    """Log tool execution using existing logger"""
    if success:
        logger.info(f"Tool {tool_name} executed successfully in {duration:.3f}s")
    else:
        logger.error(f"Tool {tool_name} failed in {duration:.3f}s with params: {params}")

def build_sql_log_message(tool_name: str, query: str, params: dict, duration: float) -> str:
    """Build SQL logging message"""
    return f"SQL Query ({tool_name}) - Duration: {duration:.3f}s\nQuery: {query}\nParams: {params}"

def extract_meeting_info(query: str) -> Dict[str, Any]:
    """Extract meeting name and year from query"""
    import re
    
    race_keywords = {
        'australian': 'Australian Grand Prix',
        'chinese': 'Chinese Grand Prix',
        'china': 'Chinese Grand Prix',
        'japanese': 'Japanese Grand Prix',
        'japan': 'Japanese Grand Prix',
        'bahrain': 'Bahrain Grand Prix',
        'saudi': 'Saudi Arabian Grand Prix',
        'saudi arabian': 'Saudi Arabian Grand Prix',
        'miami': 'Miami Grand Prix',
        'italian emilia-romagna': 'Emilia‑Romagna Grand Prix',
        'emilia romagna': 'Emilia‑Romagna Grand Prix',
        'emilia-romagna': 'Emilia‑Romagna Grand Prix',
        'imola': 'Emilia‑Romagna Grand Prix',
        'monaco': 'Monaco Grand Prix',
        'spanish': 'Spanish Grand Prix',
        'spain': 'Spanish Grand Prix',
        'canadian': 'Canadian Grand Prix',
        'canada': 'Canadian Grand Prix', 
        'austrian': 'Austrian Grand Prix',
        'austria': 'Austrian Grand Prix',
        'british': 'British Grand Prix',
        'silverstone': 'British Grand Prix',
        'great britain': 'British Grand Prix'
    }
    
    query_lower = query.lower()
    meeting_name = None
    
    for keyword, race_name in race_keywords.items():
        if keyword in query_lower:
            meeting_name = race_name
            break
    
    # Extract year (default to 2025)
    year_match = re.search(r'20\d{2}', query)
    year = int(year_match.group()) if year_match else 2025
    
    return {
        "name": meeting_name,
        "year": year
    }