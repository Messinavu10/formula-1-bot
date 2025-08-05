import base64
import io
from PIL import Image
import pandas as pd
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig


class ChartAnalyzer(BaseComponent):
    """Analyze generated charts and provide insights"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        
        # Initialize LLM for chart analysis
        api_key = config.openai_api_key or os.getenv('OPENAI_API_KEY')
        self.analysis_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=api_key,
            max_tokens=1000
        )
        
        # Chart analysis templates
        self.analysis_templates = {
            "lap_time_progression": self._get_lap_time_analysis_prompt,
            "position_progression": self._get_position_analysis_prompt,
            "sector_analysis": self._get_sector_analysis_prompt,
            "pit_stop_analysis": self._get_pit_stop_analysis_prompt,
            "tire_strategy": self._get_tire_strategy_analysis_prompt
        }
    
    def analyze_chart(self, chart_data: Dict[str, Any], chart_type: str, user_query: str) -> str:
        """Analyze a generated chart and provide insights"""
        
        try:
            self.logger.info(f"🔍 Analyzing {chart_type} chart")
            
            # Extract chart metadata
            metadata = self._extract_chart_metadata(chart_data)
            
            # Get analysis prompt based on chart type
            if chart_type in self.analysis_templates:
                analysis_prompt = self.analysis_templates[chart_type](user_query, metadata)
            else:
                analysis_prompt = self._get_generic_analysis_prompt(user_query, metadata)
            
            # Generate analysis
            messages = [SystemMessage(content=analysis_prompt)]
            response = self.analysis_llm.invoke(messages)
            analysis = response.content
            
            self.logger.info(f"✅ Chart analysis completed for {chart_type}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing chart: {e}")
            return f"I generated a {chart_type} visualization for you, but I encountered an issue analyzing it. The chart should still be visible above."
    
    def _extract_chart_metadata(self, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from chart data"""
        metadata = {
            "visualization_type": chart_data.get("visualization_type", "unknown"),
            "filename": chart_data.get("filename", ""),
            "total_laps": chart_data.get("total_laps", 0),
            "total_drivers": chart_data.get("total_drivers", 0),
            "drivers_included": chart_data.get("drivers_included", []),
            "session_key": chart_data.get("session_key", ""),
            "success": chart_data.get("success", False)
        }
        
        # Add any additional data from the chart
        if "data" in chart_data:
            metadata.update(chart_data["data"])
        
        return metadata
    
    def _get_lap_time_analysis_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Generate analysis prompt for lap time progression charts"""
        
        drivers = metadata.get("drivers_included", [])
        total_laps = metadata.get("total_laps", 0)
        
        prompt = f"""You are an expert F1 analyst. Analyze this lap time progression chart and provide insights based ONLY on what you can see in the visualization.

Chart Details:
- Type: Lap Time Progression
- Total Laps: {total_laps}
- Drivers: {', '.join(drivers) if drivers else 'Multiple drivers'}
- User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the data visible in the chart - do not make assumptions about what you cannot see
2. Describe the actual lap time trends and patterns shown
3. Compare the actual performance between drivers if multiple are shown
4. Identify any visible incidents or unusual patterns in the data
5. Point out consistency or inconsistency in lap times
6. If drivers show similar patterns, acknowledge this
7. Do NOT invent incidents or strategies that are not visible in the chart
8. Keep analysis concise but insightful (2-3 paragraphs max)
9. Focus on what the data actually shows, not what you think might have happened

IMPORTANT: Only describe what you can see in the chart. If the chart shows similar patterns between drivers, say so. Do not make up different scenarios for each driver.

Provide your analysis:"""
        
        return prompt
    
    def _get_position_analysis_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Generate analysis prompt for position progression charts"""
        
        drivers = metadata.get("drivers_included", [])
        total_laps = metadata.get("total_laps", 0)
        
        prompt = f"""You are an expert F1 analyst. Analyze this position progression chart and provide insights based ONLY on what you can see in the visualization.

Chart Details:
- Type: Position Progression
- Total Laps: {total_laps}
- Drivers: {', '.join(drivers) if drivers else 'Multiple drivers'}
- User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the data visible in the chart - do not make assumptions about what you cannot see
2. Describe the actual position changes and trends shown
3. Compare the actual position progression between drivers if multiple are shown
4. Identify any visible dramatic position gains or losses in the data
5. Point out if drivers show similar or different position patterns
6. If drivers have similar position trends, acknowledge this
7. Do NOT invent overtakes or incidents that are not visible in the chart
8. Keep analysis concise but insightful (2-3 paragraphs max)
9. Focus on what the data actually shows, not what you think might have happened

IMPORTANT: Only describe what you can see in the chart. If the chart shows similar position patterns between drivers, say so. Do not make up different scenarios for each driver.

Provide your analysis:"""
        
        return prompt
    
    def _get_sector_analysis_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Generate analysis prompt for sector analysis charts"""
        
        drivers = metadata.get("drivers_included", [])
        
        prompt = f"""You are an expert F1 analyst. Analyze this sector analysis chart and provide insights based ONLY on what you can see in the visualization.

Chart Details:
- Type: Sector Analysis
- Drivers: {', '.join(drivers) if drivers else 'Multiple drivers'}
- User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the data visible in the chart - do not make assumptions about what you cannot see
2. Describe the actual sector performance patterns shown
3. Compare the actual sector times between drivers if multiple are shown
4. Identify any visible sector strengths or weaknesses in the data
5. Point out if drivers show similar or different sector patterns
6. If drivers have similar sector performance, acknowledge this
7. Do NOT invent sector-specific incidents that are not visible in the chart
8. Keep analysis concise but insightful (2-3 paragraphs max)
9. Focus on what the data actually shows, not what you think might have happened

IMPORTANT: Only describe what you can see in the chart. If the chart shows similar sector patterns between drivers, say so. Do not make up different scenarios for each driver.

Provide your analysis:"""
        
        return prompt
    
    def _get_pit_stop_analysis_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Generate analysis prompt for pit stop analysis charts"""
        
        drivers = metadata.get("drivers_included", [])
        
        prompt = f"""You are an expert F1 analyst. Analyze this pit stop analysis chart and provide insights based ONLY on what you can see in the visualization.

Chart Details:
- Type: Pit Stop Analysis
- Drivers: {', '.join(drivers) if drivers else 'Multiple drivers'}
- User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the data visible in the chart - do not make assumptions about what you cannot see
2. Describe the actual pit stop timing and duration shown
3. Compare the actual pit stop performance between drivers if multiple are shown
4. Identify any visible fastest or slowest pit stops in the data
5. Point out if drivers show similar or different pit stop patterns
6. If drivers have similar pit stop strategies, acknowledge this
7. Do NOT invent pit stop errors or incidents that are not visible in the chart
8. Keep analysis concise but insightful (2-3 paragraphs max)
9. Focus on what the data actually shows, not what you think might have happened

IMPORTANT: Only describe what you can see in the chart. If the chart shows similar pit stop patterns between drivers, say so. Do not make up different scenarios for each driver.

Provide your analysis:"""
        
        return prompt
    
    def _get_tire_strategy_analysis_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Generate analysis prompt for tire strategy charts"""
        
        drivers = metadata.get("drivers_included", [])
        
        prompt = f"""You are an expert F1 analyst. Analyze this tire strategy chart and provide insights based ONLY on what you can see in the visualization.

Chart Details:
- Type: Tire Strategy Analysis
- Drivers: {', '.join(drivers) if drivers else 'Multiple drivers'}
- User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the data visible in the chart - do not make assumptions about what you cannot see
2. Describe the actual tire compounds used (Medium, Hard, Soft) as shown in the legend
3. Compare the actual stint lengths and timing between drivers
4. Identify if drivers used similar or different strategies
5. Point out any notable differences in timing or compound choices
6. If drivers used identical strategies, acknowledge this
7. Do NOT invent strategies that are not shown in the chart
8. Keep analysis concise but insightful (2-3 paragraphs max)
9. Focus on what the data actually shows, not what you think might have happened

IMPORTANT: Only describe what you can see in the chart. If the chart shows both drivers using the same strategy, say so. Do not make up different strategies for each driver.

Provide your analysis:"""
        
        return prompt
    
    def _get_generic_analysis_prompt(self, user_query: str, metadata: Dict[str, Any]) -> str:
        """Generate generic analysis prompt for unknown chart types"""
        
        chart_type = metadata.get("visualization_type", "data visualization")
        
        prompt = f"""You are an expert F1 analyst. Analyze this {chart_type} chart and provide insights based ONLY on what you can see in the visualization.

Chart Details:
- Type: {chart_type}
- User Query: "{user_query}"

CRITICAL INSTRUCTIONS:
1. Analyze ONLY the data visible in the chart - do not make assumptions about what you cannot see
2. Describe the actual patterns and trends shown in the data
3. Compare the actual performance between entities if multiple are shown
4. Identify any visible notable performances or events in the data
5. Point out if entities show similar or different patterns
6. If entities have similar performance, acknowledge this
7. Do NOT invent incidents or scenarios that are not visible in the chart
8. Keep analysis concise but insightful (2-3 paragraphs max)
9. Focus on what the data actually shows, not what you think might have happened

IMPORTANT: Only describe what you can see in the chart. If the chart shows similar patterns between entities, say so. Do not make up different scenarios for each entity.

Provide your analysis:"""
        
        return prompt 