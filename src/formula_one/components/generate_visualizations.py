import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import pandas as pd
import numpy as np
import os
import uuid
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

from src.formula_one.components.base_component import BaseComponent
from src.formula_one.entity.config_entity import DatabaseConfig
from src.formula_one.entity.mcp_config_entity import ToolResult

class VisualizationGenerator(BaseComponent):
    """Generate F1 data visualizations using Plotly"""
    
    def __init__(self, config, db_config: DatabaseConfig):
        super().__init__(config, db_config)
        
        # Create visualization storage directory
        self.viz_dir = Path("artifacts/visualizations")
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # F1 color scheme
        self.f1_colors = {
            'primary': '#e10600',      # F1 Red
            'secondary': '#ff6b35',    # F1 Orange
            'dark': '#1a1a1a',         # F1 Dark
            'light': '#3a3a3a',        # F1 Light Gray
            'success': '#00d4aa',      # F1 Green
            'warning': '#ffd700',      # F1 Yellow
            'teams': {
                'Red Bull Racing': '#3671C6',
                'Ferrari': '#F91536',
                'McLaren': '#FF8700',
                'Mercedes': '#6CD3BF',
                'Aston Martin': '#358C75',
                'Alpine': '#2293D1',
                'Williams': '#37BEDD',
                'Haas': '#B6BABD',
                'Racing Bulls': '#5E8FAA',
                'Sauber': '#52E252'
            }
        }
        
        # Clean up old visualizations (older than 24 hours)
        self._cleanup_old_visualizations()
    
    def _cleanup_old_visualizations(self):
        """Clean up visualizations older than 24 hours"""
        try:
            current_time = datetime.now()
            for file_path in self.viz_dir.glob("*.html"):
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > 86400:  # 24 hours
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old visualization: {file_path}")
        except Exception as e:
            self.logger.warning(f"Error cleaning up old visualizations: {e}")
    
    def _generate_filename(self, viz_type: str) -> Tuple[str, str]:
        """Generate unique filename for visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"f1_{viz_type}_{timestamp}_{unique_id}.html"
        filepath = self.viz_dir / filename
        return str(filepath), filename
    
    def _save_visualization(self, fig: go.Figure, filepath: str) -> str:
        """Save visualization and return base64 encoded image"""
        try:
            # Save as HTML
            fig.write_html(filepath, include_plotlyjs='cdn')
            
            # Convert to base64 for ML model consumption
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            self.logger.info(f"Visualization saved: {filepath}")
            return img_base64
            
        except Exception as e:
            self.logger.error(f"Error saving visualization: {e}")
            raise
        
    def create_lap_time_progression(self, session_key: str, driver_filter: str = None, team_filter: str = None) -> ToolResult:
        """Create lap time progression chart"""
        start_time = datetime.now()
        
        try:
            # Get query from query builder
            query = self.query_builder.build_lap_time_progression_query(session_key, driver_filter, team_filter)
            
            # Build parameters
            params = {"session_key": session_key}
            
            if driver_filter:
                # Handle multiple drivers
                if "," in driver_filter:
                    driver_names = [name.strip() for name in driver_filter.split(",")]
                    for i, driver in enumerate(driver_names):
                        params[f"driver_filter_{i}"] = driver
                else:
                    params["driver_filter"] = driver_filter
            
            if team_filter:
                # Handle multiple teams
                if "," in team_filter:
                    team_names = [name.strip() for name in team_filter.split(",")]
                    for i, team in enumerate(team_names):
                        params[f"team_filter_{i}"] = team
                else:
                    params["team_filter"] = team_filter
            
            # Execute query
            data = self.db_utils.execute_mcp_query(query, params)
            
            if not data:
                return ToolResult(
                    success=False,
                    data={},
                    error="No lap time data found for this session",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'lap_number', 'driver_name', 'team_name', 'lap_duration',
                'sector1', 'sector2', 'sector3', 'had_incident', 'safety_car_lap'
            ])
            
            # Create visualization
            fig = go.Figure()
            
            # Add traces for each driver
            for driver in df['driver_name'].unique():
                driver_data = df[df['driver_name'] == driver]
                team_name = driver_data['team_name'].iloc[0]
                team_color = self.f1_colors['teams'].get(team_name, self.f1_colors['primary'])
                
                # Main lap time line
                fig.add_trace(go.Scatter(
                    x=driver_data['lap_number'],
                    y=driver_data['lap_duration'],
                    mode='lines+markers',
                    name=f"{driver} ({team_name})",
                    line=dict(color=team_color, width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Lap: %{x}<br>' +
                                'Time: %{y:.3f}s<br>' +
                                '<extra></extra>'
                ))
                
                # Highlight incidents
                incidents = driver_data[driver_data['had_incident'] == True]
                if not incidents.empty:
                    fig.add_trace(go.Scatter(
                        x=incidents['lap_number'],
                        y=incidents['lap_duration'],
                        mode='markers',
                        name=f"{driver} - Incidents",
                        marker=dict(
                            symbol='x',
                            size=10,
                            color='red',
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False,
                        hovertemplate='<b>INCIDENT</b><br>' +
                                    'Lap: %{x}<br>' +
                                    'Time: %{y:.3f}s<br>' +
                                    '<extra></extra>'
                    ))
            
            # Update layout
            fig.update_layout(
                title=f"Lap Time Progression",
                xaxis_title="Lap Number",
                yaxis_title="Lap Time (seconds)",
                template="plotly_dark",
                plot_bgcolor=self.f1_colors['dark'],
                paper_bgcolor=self.f1_colors['dark'],
                font=dict(color='white'),
                hovermode='closest',
                legend=dict(
                    bgcolor=self.f1_colors['light'],
                    bordercolor=self.f1_colors['primary'],
                    borderwidth=1
                )
            )
            
            # Save visualization
            filepath, filename = self._generate_filename("lap_progression")
            img_base64 = self._save_visualization(fig, filepath)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "visualization_type": "lap_time_progression",
                    "filename": filename,
                    "filepath": filepath,
                    "image_base64": img_base64,
                    "session_key": session_key,
                    "total_laps": len(df['lap_number'].unique()),
                    "total_drivers": len(df['driver_name'].unique()),
                    "drivers_included": df['driver_name'].unique().tolist()
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error creating lap time progression: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def create_position_progression(self, session_key: str, driver_filter: str = None, team_filter: str = None) -> ToolResult:
        """Create position progression chart using positions_transformed data"""
        start_time = datetime.now()
        
        try:
            # Get query from query builder with proper filtering
            query = self.query_builder.build_position_progression_query(session_key, driver_filter, team_filter)
            
            # Build parameters
            params = {"session_key": session_key}
            
            if driver_filter:
                # Handle multiple drivers
                if "," in driver_filter:
                    driver_names = [name.strip() for name in driver_filter.split(",")]
                    for i, driver in enumerate(driver_names):
                        params[f"driver_filter_{i}"] = driver
                else:
                    params["driver_filter_0"] = driver_filter
            
            if team_filter:
                # Handle multiple teams
                if "," in team_filter:
                    team_names = [name.strip() for name in team_filter.split(",")]
                    for i, team in enumerate(team_names):
                        params[f"team_filter_{i}"] = team
                else:
                    params["team_filter_0"] = team_filter
            
            # Debug logging
            self.logger.info(f"Position progression query with filters - driver_filter: {driver_filter}, team_filter: {team_filter}")
            self.logger.info(f"Query parameters: {params}")
            
            # Execute query
            data = self.db_utils.execute_mcp_query(query, params)
            
            if not data:
                return ToolResult(
                    success=False,
                    data={},
                    error="No position data found for this session",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Convert to DataFrame with new column structure
            df = pd.DataFrame(data, columns=[
                'timestamp', 'driver_number', 'driver_name', 'team_name', 
                'position', 'position_change', 'is_leader', 'position_sequence'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Debug logging
            self.logger.info(f"Retrieved {len(df)} rows of position data")
            self.logger.info(f"Unique drivers in data: {df['driver_name'].unique().tolist()}")
            
            if df.empty:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"No position data found for the specified filters",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Create visualization
            fig = go.Figure()
            
            # Add traces for each driver
            for driver in df['driver_name'].unique():
                driver_data = df[df['driver_name'] == driver].sort_values('timestamp')
                team_name = driver_data['team_name'].iloc[0]
                team_color = self.f1_colors['teams'].get(team_name, self.f1_colors['primary'])
                
                # Only add trace if we have meaningful position changes
                if len(driver_data) > 1:
                    # Create hover text with position change information
                    hover_text = []
                    for _, row in driver_data.iterrows():
                        change_text = ""
                        if row['position_change'] != 0:
                            if row['position_change'] > 0:
                                change_text = f" (+{row['position_change']})"
                            else:
                                change_text = f" ({row['position_change']})"
                        
                        leader_text = " ��" if row['is_leader'] else ""
                        hover_text.append(f"{row['position']}{change_text}{leader_text}")
                    
                    fig.add_trace(go.Scatter(
                        x=driver_data['timestamp'],
                        y=driver_data['position'],
                        mode='lines+markers',
                        name=f"{driver} ({team_name})",
                        line=dict(color=team_color, width=3),
                        marker=dict(size=6),
                        text=hover_text,
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Time: %{x}<br>' +
                                    'Position: %{text}<br>' +
                                    '<extra></extra>'
                    ))
            
            # Update layout (inverted Y-axis for positions)
            title = f"Position Progression"
            if driver_filter:
                title += f" - {driver_filter}"
            elif team_filter:
                title += f" - {team_filter}"
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Position",
                template="plotly_dark",
                plot_bgcolor=self.f1_colors['dark'],
                paper_bgcolor=self.f1_colors['dark'],
                font=dict(color='white', size=14),
                height=600,
                width=1200,
                yaxis=dict(
                    autorange='reversed',  # Position 1 at top
                    tickmode='linear',
                    tick0=1,
                    dtick=1,
                    range=[1, max(df['position'].max(), 20)]  # Ensure proper range
                ),
                hovermode='closest',
                legend=dict(
                    bgcolor=self.f1_colors['light'],
                    bordercolor=self.f1_colors['primary'],
                    borderwidth=1
                )
            )
            
            # Save visualization
            filepath, filename = self._generate_filename("position_progression")
            img_base64 = self._save_visualization(fig, filepath)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "visualization_type": "position_progression",
                    "filename": filename,
                    "filepath": filepath,
                    "image_base64": img_base64,
                    "session_key": session_key,
                    "total_position_changes": len(df),
                    "total_drivers": len(df['driver_name'].unique()),
                    "drivers_included": df['driver_name'].unique().tolist(),
                    "time_range": {
                        "start": df['timestamp'].min().isoformat(),
                        "end": df['timestamp'].max().isoformat()
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error creating position progression: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
    def create_sector_analysis(self, session_key: str, driver_filter: str = None, team_filter: str = None) -> ToolResult:
        """Create sector analysis visualization with subplots"""
        start_time = datetime.now()
        
        try:
            # Get query from query builder
            query = self.query_builder.build_sector_analysis_viz_query(session_key, driver_filter, team_filter)
            
            # Build parameters
            params = {"session_key": session_key}
            if driver_filter:
                # Handle multiple drivers
                if "," in driver_filter:
                    driver_names = [name.strip() for name in driver_filter.split(",")]
                    for i, driver in enumerate(driver_names):
                        params[f"driver_filter_{i}"] = driver
                else:
                    params["driver_filter"] = driver_filter
            
            if team_filter:
                # Handle multiple teams
                if "," in team_filter:
                    team_names = [name.strip() for name in team_filter.split(",")]
                    for i, team in enumerate(team_names):
                        params[f"team_filter_{i}"] = team
                else:
                    params["team_filter"] = team_filter
            
            # Execute query
            data = self.db_utils.execute_mcp_query(query, params)
            
            if not data:
                return ToolResult(
                    success=False,
                    data={},
                    error="No sector data found for this session",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'driver_name', 'team_name', 'sector1', 'sector2', 'sector3', 
                'lap_duration', 'lap_number'
            ])
            
            if df.empty:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"No sector data found for the specified filters",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Create subplots
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=('Sector 1 Times', 'Sector 2 Times', 'Sector 3 Times', 'Lap Times'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add traces for each driver
            for driver in df['driver_name'].unique():
                driver_data = df[df['driver_name'] == driver]
                team_name = driver_data['team_name'].iloc[0]
                team_color = self.f1_colors['teams'].get(team_name, self.f1_colors['primary'])
                
                # Sector 1
                fig.add_trace(
                    go.Scatter(
                        x=driver_data['lap_number'],
                        y=driver_data['sector1'],
                        mode='lines+markers',
                        name=f"{driver} ({team_name})",
                        line=dict(color=team_color, width=2),
                        marker=dict(size=3),
                        legendgroup=driver,  # Group all traces for this driver
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Sector 2
                fig.add_trace(
                    go.Scatter(
                        x=driver_data['lap_number'],
                        y=driver_data['sector2'],
                        mode='lines+markers',
                        name=f"{driver} ({team_name})",
                        line=dict(color=team_color, width=2),
                        marker=dict(size=3),
                        legendgroup=driver,  # Group all traces for this driver
                        showlegend=False  # Only show in legend once
                    ),
                    row=1, col=2
                )
                
                # Sector 3
                fig.add_trace(
                    go.Scatter(
                        x=driver_data['lap_number'],
                        y=driver_data['sector3'],
                        mode='lines+markers',
                        name=f"{driver} ({team_name})",
                        line=dict(color=team_color, width=2),
                        marker=dict(size=3),
                        legendgroup=driver,  # Group all traces for this driver
                        showlegend=False  # Only show in legend once
                    ),
                    row=2, col=1
                )
                
                # Lap times
                fig.add_trace(
                    go.Scatter(
                        x=driver_data['lap_number'],
                        y=driver_data['lap_duration'],
                        mode='lines+markers',
                        name=f"{driver} ({team_name})",
                        line=dict(color=team_color, width=2),
                        marker=dict(size=3),
                        legendgroup=driver,  # Group all traces for this driver
                        showlegend=False  # Only show in legend once
                    ),
                    row=2, col=2
                )
            
            # Update layout
            title = f"Sector Analysis"
            if driver_filter:
                title += f" - {driver_filter}"
            elif team_filter:
                title += f" - {team_filter}"
            
            fig.update_layout(
                title=title,
                template="plotly_dark",
                plot_bgcolor=self.f1_colors['dark'],
                paper_bgcolor=self.f1_colors['dark'],
                font=dict(color='white'),
                height=800,
                showlegend=True,
                legend=dict(
                    bgcolor=self.f1_colors['light'],
                    bordercolor=self.f1_colors['primary'],
                    borderwidth=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Lap Number", row=1, col=1)
            fig.update_xaxes(title_text="Lap Number", row=1, col=2)
            fig.update_xaxes(title_text="Lap Number", row=2, col=1)
            fig.update_xaxes(title_text="Lap Number", row=2, col=2)
            
            fig.update_yaxes(title_text="Sector 1 Time (s)", row=1, col=1)
            fig.update_yaxes(title_text="Sector 2 Time (s)", row=1, col=2)
            fig.update_yaxes(title_text="Sector 3 Time (s)", row=2, col=1)
            fig.update_yaxes(title_text="Lap Time (s)", row=2, col=2)
            
            # Save visualization
            filepath, filename = self._generate_filename("sector_analysis")
            img_base64 = self._save_visualization(fig, filepath)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "visualization_type": "sector_analysis",
                    "filename": filename,
                    "filepath": filepath,
                    "image_base64": img_base64,
                    "session_key": session_key,
                    "total_drivers": len(df['driver_name'].unique()),
                    "drivers_included": df['driver_name'].unique().tolist()
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error creating sector analysis: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        
    def create_pit_stop_analysis(self, session_key: str, driver_filter: str = None, team_filter: str = None, 
                            analysis_type: str = "comprehensive") -> ToolResult:
        """Create pit stop analysis visualization with intelligent graph selection"""
        start_time = datetime.now()
        
        try:
            # Get query from query builder
            query = self.query_builder.build_pit_stop_analysis_query(session_key, driver_filter, team_filter)
            
            # Build parameters
            params = {"session_key": session_key}
            
            if driver_filter:
                # Handle multiple drivers
                if "," in driver_filter:
                    driver_names = [name.strip() for name in driver_filter.split(",")]
                    for i, driver in enumerate(driver_names):
                        params[f"driver_filter_{i}"] = driver
                else:
                    params["driver_filter"] = driver_filter
            
            if team_filter:
                # Handle multiple teams
                if "," in team_filter:
                    team_names = [name.strip() for name in team_filter.split(",")]
                    for i, team in enumerate(team_names):
                        params[f"team_filter_{i}"] = team
                else:
                    params["team_filter"] = team_filter
            
            # Execute query
            data = self.db_utils.execute_mcp_query(query, params)
            
            if not data:
                return ToolResult(
                    success=False,
                    data={},
                    error="No pit stop data found for this session",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'driver_name', 'team_name', 'lap_number', 'pit_duration', 
                'pit_stop_count', 'pit_stop_timing', 'normal_pit_stop', 
                'long_pit_stop', 'penalty_pit_stop'
            ])
            
            if df.empty:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"No pit stop data found for the specified filters",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Determine which graphs to show based on context
            num_drivers = len(df['driver_name'].unique())
            num_teams = len(df['team_name'].unique())
            
            # Default to comprehensive analysis, but can be overridden
            if analysis_type == "comprehensive":
                # Show all graphs for comprehensive analysis
                show_duration = True
                show_distribution = True
                show_performance = True
                show_timing = True
            elif analysis_type == "simple" or (num_drivers == 1 and num_teams == 1):
                # For single driver/team, show only the most relevant graphs
                show_duration = True
                show_distribution = False  # Not meaningful for single driver
                show_performance = True
                show_timing = True
            elif analysis_type == "comparison" or (num_drivers > 1 or num_teams > 1):
                # For comparisons, focus on comparison-relevant graphs
                show_duration = True
                show_distribution = True
                show_performance = True
                show_timing = False  # Less relevant for comparisons
            else:
                # Default fallback
                show_duration = True
                show_distribution = True
                show_performance = True
                show_timing = True
            
            # Create individual figures for each analysis type
            figures = []
            figure_titles = []
            
            # 1. Pit Stop Duration by Lap (always relevant)
            if show_duration:
                fig1 = go.Figure()
                for driver in df['driver_name'].unique():
                    driver_data = df[df['driver_name'] == driver]
                    team_name = driver_data['team_name'].iloc[0]
                    team_color = self.f1_colors['teams'].get(team_name, self.f1_colors['primary'])
                    
                    fig1.add_trace(go.Scatter(
                        x=driver_data['lap_number'],
                        y=driver_data['pit_duration'],
                        mode='lines+markers',
                        name=f"{driver} ({team_name})",
                        line=dict(color=team_color, width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Lap: %{x}<br>' +
                                    'Duration: %{y:.2f}s<br>' +
                                    '<extra></extra>'
                    ))
                
                title = f"Pit Stop Duration by Lap"
                if driver_filter:
                    title += f" - {driver_filter}"
                elif team_filter:
                    title += f" - {team_filter}"
                
                fig1.update_layout(
                    title=title,
                    xaxis_title="Lap Number",
                    yaxis_title="Pit Duration (seconds)",
                    template="plotly_dark",
                    plot_bgcolor=self.f1_colors['dark'],
                    paper_bgcolor=self.f1_colors['dark'],
                    font=dict(color='white', size=14),
                    height=600,
                    width=1200,
                    hovermode='closest',
                    legend=dict(
                        bgcolor=self.f1_colors['light'],
                        bordercolor=self.f1_colors['primary'],
                        borderwidth=1
                    )
                )
                figures.append(fig1)
                figure_titles.append("Pit Stop Duration by Lap")
            
            # 2. Pit Stop Types Distribution (only for multiple drivers or comprehensive)
            if show_distribution and num_drivers > 1:
                fig2 = go.Figure()
                
                # Prepare data for stacked bar chart
                drivers = df['driver_name'].unique()
                normal_stops = []
                long_stops = []
                penalty_stops = []
                
                for driver in drivers:
                    driver_data = df[df['driver_name'] == driver]
                    normal_stops.append(len(driver_data[driver_data['normal_pit_stop'] == True]))
                    long_stops.append(len(driver_data[driver_data['long_pit_stop'] == True]))
                    penalty_stops.append(len(driver_data[driver_data['penalty_pit_stop'] == True]))
                
                # Add traces for each pit stop type
                fig2.add_trace(go.Bar(
                    x=drivers,
                    y=normal_stops,
                    name='Normal Pit Stops',
                    marker_color='#00d4aa',
                    hovertemplate='<b>%{x}</b><br>Normal Stops: %{y}<br><extra></extra>'
                ))
                
                fig2.add_trace(go.Bar(
                    x=drivers,
                    y=long_stops,
                    name='Long Pit Stops',
                    marker_color='#ff6b35',
                    hovertemplate='<b>%{x}</b><br>Long Stops: %{y}<br><extra></extra>'
                ))
                
                fig2.add_trace(go.Bar(
                    x=drivers,
                    y=penalty_stops,
                    name='Penalty Pit Stops',
                    marker_color='#e10600',
                    hovertemplate='<b>%{x}</b><br>Penalty Stops: %{y}<br><extra></extra>'
                ))
                
                title = f"Pit Stop Types Distribution"
                if driver_filter:
                    title += f" - {driver_filter}"
                elif team_filter:
                    title += f" - {team_filter}"
                
                fig2.update_layout(
                    title=title,
                    xaxis_title="Driver",
                    yaxis_title="Number of Pit Stops",
                    template="plotly_dark",
                    plot_bgcolor=self.f1_colors['dark'],
                    paper_bgcolor=self.f1_colors['dark'],
                    font=dict(color='white', size=14),
                    height=600,
                    width=1200,
                    barmode='stack',
                    legend=dict(
                        bgcolor=self.f1_colors['light'],
                        bordercolor=self.f1_colors['primary'],
                        borderwidth=1
                    )
                )
                figures.append(fig2)
                figure_titles.append("Pit Stop Types Distribution")
            
            # 3. Pit Stop Performance Summary (Box Plot) - always relevant
            if show_performance:
                fig3 = go.Figure()
                
                for driver in df['driver_name'].unique():
                    driver_data = df[df['driver_name'] == driver]
                    team_name = driver_data['team_name'].iloc[0]
                    team_color = self.f1_colors['teams'].get(team_name, self.f1_colors['primary'])
                    
                    fig3.add_trace(go.Box(
                        y=driver_data['pit_duration'],
                        name=f"{driver} ({team_name})",
                        marker_color=team_color,
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Duration: %{y:.2f}s<br>' +
                                    '<extra></extra>'
                    ))
                
                title = f"Pit Stop Performance Summary"
                if driver_filter:
                    title += f" - {driver_filter}"
                elif team_filter:
                    title += f" - {team_filter}"
                
                fig3.update_layout(
                    title=title,
                    xaxis_title="Driver",
                    yaxis_title="Pit Duration (seconds)",
                    template="plotly_dark",
                    plot_bgcolor=self.f1_colors['dark'],
                    paper_bgcolor=self.f1_colors['dark'],
                    font=dict(color='white', size=14),
                    height=600,
                    width=1200,
                    showlegend=False,
                    boxmode='group'
                )
                figures.append(fig3)
                figure_titles.append("Pit Stop Performance Summary")
            
            # 4. Pit Stop Timing Analysis (if timing data exists and relevant)
            if show_timing and 'pit_stop_timing' in df.columns and not df['pit_stop_timing'].isna().all():
                fig4 = go.Figure()
                
                for driver in df['driver_name'].unique():
                    driver_data = df[df['driver_name'] == driver]
                    team_name = driver_data['team_name'].iloc[0]
                    team_color = self.f1_colors['teams'].get(team_name, self.f1_colors['primary'])
                    
                    timing_data = driver_data[driver_data['pit_stop_timing'].notna()]
                    if not timing_data.empty:
                        fig4.add_trace(go.Scatter(
                            x=timing_data['lap_number'],
                            y=timing_data['pit_stop_timing'],
                            mode='markers+lines',
                            name=f"{driver} ({team_name})",
                            line=dict(color=team_color, width=2),
                            marker=dict(
                                color=team_color,
                                size=10,
                                symbol='diamond'
                            ),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Lap: %{x}<br>' +
                                        'Timing: %{y}<br>' +
                                        '<extra></extra>'
                        ))
                
                title = f"Pit Stop Timing Analysis"
                if driver_filter:
                    title += f" - {driver_filter}"
                elif team_filter:
                    title += f" - {team_filter}"
                
                fig4.update_layout(
                    title=title,
                    xaxis_title="Lap Number",
                    yaxis_title="Pit Stop Timing",
                    template="plotly_dark",
                    plot_bgcolor=self.f1_colors['dark'],
                    paper_bgcolor=self.f1_colors['dark'],
                    font=dict(color='white', size=14),
                    height=600,
                    width=1200,
                    hovermode='closest',
                    legend=dict(
                        bgcolor=self.f1_colors['light'],
                        bordercolor=self.f1_colors['primary'],
                        borderwidth=1
                    )
                )
                figures.append(fig4)
                figure_titles.append("Pit Stop Timing Analysis")
            
            # Create a combined HTML with scrollable layout
            session_key_str = str(session_key)
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pit Stop Analysis - Session {session_key_str}</title>
                <style>
                    body {{
                        background-color: #1a1a1a;
                        color: white;
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                    }}
                    .chart-container {{
                        margin-bottom: 40px;
                        background-color: #3a3a3a;
                        border-radius: 10px;
                        padding: 20px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    }}
                    h1 {{
                        text-align: center;
                        color: #e10600;
                        margin-bottom: 30px;
                    }}
                    .summary {{
                        background-color: #2a2a2a;
                        border-radius: 8px;
                        padding: 15px;
                        margin-bottom: 20px;
                        border-left: 4px solid #e10600;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Pit Stop Analysis - Session {session_key_str}</h1>
                    <div class="summary">
                        <strong>Analysis Summary:</strong><br>
                        • Total Pit Stops: {len(df)}<br>
                        • Drivers Analyzed: {num_drivers}<br>
                        • Teams Analyzed: {num_teams}<br>
                        • Average Duration: {df['pit_duration'].mean():.2f}s<br>
                        • Fastest Stop: {df['pit_duration'].min():.2f}s<br>
                        • Slowest Stop: {df['pit_duration'].max():.2f}s
                    </div>
            """
            
            # Add each figure to the HTML
            for i, fig in enumerate(figures):
                html_content += f'<div class="chart-container">{fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)}</div>'
            
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Save the combined visualization
            filepath, filename = self._generate_filename("pit_stop_analysis")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Convert the first figure to base64 for ML model consumption
            img_base64 = base64.b64encode(figures[0].to_image(format="png", width=1200, height=800)).decode('utf-8')
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "visualization_type": "pit_stop_analysis",
                    "filename": filename,
                    "filepath": filepath,
                    "image_base64": img_base64,
                    "session_key": session_key,
                    "total_pit_stops": len(df),
                    "total_drivers": len(df['driver_name'].unique()),
                    "drivers_included": df['driver_name'].unique().tolist(),
                    "graphs_shown": figure_titles,
                    "analysis_type": analysis_type,
                    "analysis_summary": {
                        "normal_stops": len(df[df['normal_pit_stop'] == True]),
                        "long_stops": len(df[df['long_pit_stop'] == True]),
                        "penalty_stops": len(df[df['penalty_pit_stop'] == True]),
                        "avg_duration": df['pit_duration'].mean(),
                        "fastest_stop": df['pit_duration'].min(),
                        "slowest_stop": df['pit_duration'].max()
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error creating pit stop analysis: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    
    def create_tire_strategy_visualization(self, session_key: str, driver_filter: str = None, team_filter: str = None) -> ToolResult:
        """Create tire strategy visualization as a clear Gantt chart"""
        start_time = datetime.now()
        
        try:
            # Get query from query builder
            query = self.query_builder.build_tire_strategy_viz_query(session_key, driver_filter, team_filter)
            
            # Build parameters
            params = {"session_key": session_key}
            
            if driver_filter:
                # Handle multiple drivers
                if "," in driver_filter:
                    driver_names = [name.strip() for name in driver_filter.split(",")]
                    for i, driver in enumerate(driver_names):
                        params[f"driver_filter_{i}"] = driver
                else:
                    params["driver_filter"] = driver_filter
            
            if team_filter:
                # Handle multiple teams
                if "," in team_filter:
                    team_names = [name.strip() for name in team_filter.split(",")]
                    for i, team in enumerate(team_names):
                        params[f"team_filter_{i}"] = team
                else:
                    params["team_filter"] = team_filter
            
            data = self.db_utils.execute_mcp_query(query, params)
            
            if not data:
                return ToolResult(
                    success=False,
                    data={},
                    error="No tire strategy data found for this session",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'driver_name', 'team_name', 'compound', 'lap_start', 'lap_end', 'stint_length'
            ])
            
            if df.empty:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"No tire strategy data found for the specified filters",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Get total race length
            total_laps = int(df['lap_end'].max())
            num_drivers = len(df['driver_name'].unique())
            
            # Create figure
            fig = go.Figure()
            
            # Color mapping for compounds with better contrast
            compound_colors = {
                'soft': '#FF4136',      # Bright red
                'medium': '#FFDC00',    # Bright yellow
                'hard': '#AAAAAA',      # Light gray
                'intermediate': '#2ECC40',  # Green
                'wet': '#0074D9',       # Blue
            }
            
            # Track compounds for legend
            added_compounds = set()
            
            # Sort drivers for consistent ordering
            drivers_sorted = sorted(df['driver_name'].unique())
            
            # Add traces for each driver's stints
            for driver in drivers_sorted:
                driver_data = df[df['driver_name'] == driver].sort_values('lap_start')
                team_name = driver_data['team_name'].iloc[0]
                
                for _, stint in driver_data.iterrows():
                    compound = stint['compound'].lower() if stint['compound'] else 'unknown'
                    color = compound_colors.get(compound, '#808080')
                    
                    # Only show legend for each compound once
                    show_legend = compound not in added_compounds
                    if show_legend:
                        added_compounds.add(compound)
                    
                    # Create horizontal bar for this stint with rounded corners
                    fig.add_trace(go.Bar(
                        x=[stint['stint_length']],  # Bar width
                        y=[f"{driver} ({team_name})"],
                        orientation='h',
                        name=compound.title(),
                        marker_color=color,
                        marker=dict(
                            line=dict(width=0),  # Remove border
                            cornerradius=8  # Add rounded corners
                        ),
                        showlegend=show_legend,
                        base=stint['lap_start'] - 1,  # Start position
                        width=0.7,  # Slightly thinner bars for better spacing
                        hovertemplate=f'<b>{driver} ({team_name})</b><br>' +
                                    f'Compound: {stint["compound"]}<br>' +
                                    f'Laps: {stint["lap_start"]}-{stint["lap_end"]}<br>' +
                                    f'Duration: {stint["stint_length"]} laps<br>' +
                                    '<extra></extra>'
                    ))
            
            # Update layout for clear Gantt chart with better fitting
            title = f"Tire Strategy - ({total_laps} laps)"
            if driver_filter:
                title += f" - {driver_filter}"
            elif team_filter:
                title += f" - {team_filter}"
            
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=16, color='white')
                ),
                template="plotly_dark",
                plot_bgcolor=self.f1_colors['dark'],
                paper_bgcolor=self.f1_colors['dark'],
                font=dict(color='white', size=12),
                height=max(600, num_drivers * 40),  # Adjusted height calculation
                width=1200,  # Slightly narrower for better fit
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(0,0,0,0.8)',
                    bordercolor='white',
                    borderwidth=1,
                    title=dict(text="Tire Compounds", font=dict(color='white')),
                    font=dict(color='white'),
                    x=1.02,
                    y=1,
                    xanchor='left',
                    yanchor='top'
                ),
                margin=dict(l=180, r=180, t=60, b=60),  # Adjusted margins
                bargap=0.15,  # Increased gap between bars
                bargroupgap=0.1  # Increased gap between bar groups
            )
            
            # Update axes separately with better range fitting
            fig.update_xaxes(
                title="Lap Number",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.2)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.5)',
                range=[-1, total_laps + 1],  # Add padding to show full range
                tickmode='linear',
                tick0=0,
                dtick=5,
                tickfont=dict(color='white'),
                tickangle=0
            )
            
            fig.update_yaxes(
                title="Driver",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                categoryorder='array',
                categoryarray=[f"{driver} ({df[df['driver_name'] == driver]['team_name'].iloc[0]})" 
                            for driver in drivers_sorted],
                tickfont=dict(color='white', size=11)
            )
            
            # Save visualization
            filepath, filename = self._generate_filename("tire_strategy")
            img_base64 = self._save_visualization(fig, filepath)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "visualization_type": "tire_strategy",
                    "filename": filename,
                    "filepath": filepath,
                    "image_base64": img_base64,
                    "session_key": session_key,
                    "total_stints": int(len(df)),
                    "total_drivers": int(num_drivers),
                    "total_laps": total_laps,
                    "drivers_included": drivers_sorted
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error creating tire strategy visualization: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )