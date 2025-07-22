class QueryBuilder:
    """Class to build SQL queries for MCP tools with parameterized placeholders"""

    def build_meeting_query(self, event_name, year):
        """Build query to get meeting key for a specific event"""
        return """
        SELECT meeting_key, meeting_name, country_name, date_start, year
        FROM meetings 
        WHERE LOWER(meeting_name) LIKE LOWER(:event_name)
        AND year = :year
        ORDER BY date_start DESC
        LIMIT 1
        """

    def build_session_query(self, meeting_key, session_type):
        """Build query to get session key for a specific session type with enhanced logic"""
        # Base query for session lookup
        base_query = """
        SELECT session_key, session_name, session_type, date_start, date_end
        FROM sessions_transformed 
        WHERE meeting_key = :meeting_key
        """
        
        # Handle Qualifying case with special logic
        if session_type.lower() in ["qualifying", "qual"]:
            return """
            SELECT session_key, session_name, session_type, date_start, date_end
            FROM sessions_transformed 
            WHERE meeting_key = :meeting_key 
            AND session_name LIKE :session_type_pattern
            ORDER BY date_start ASC
            LIMIT 1
            """
        else:
            return """
            SELECT session_key, session_name, session_type, date_start, date_end
            FROM sessions_transformed 
            WHERE meeting_key = :meeting_key 
            AND UPPER(session_name) LIKE UPPER(:session_type)
            ORDER BY date_start ASC
            LIMIT 1
            """

    def build_fastest_lap_query(self, session_key, driver_filter=None, team_filter=None):
        """Build query to get fastest lap time and details"""
        where_conditions = ["l.session_key = :session_key"]
        params = {"session_key": session_key}
        
        if driver_filter:
            where_conditions.append("UPPER(d.full_name) = UPPER(:driver_filter)")
            params["driver_filter"] = driver_filter
        if team_filter:
            where_conditions.append("UPPER(d.team_name) = UPPER(:team_filter)")
            params["team_filter"] = team_filter
        
        where_clause = " AND ".join(where_conditions)
        
        return f"""
        SELECT 
            d.full_name,
            d.team_name,
            l.lap_number,
            l.lap_duration,
            l.duration_sector_1,
            l.duration_sector_2,
            l.duration_sector_3,
            l.had_incident,
            l.safety_car_lap,
            l.is_outlier
        FROM laps_transformed l
        JOIN drivers_transformed d ON l.driver_number = d.driver_number 
            AND l.meeting_key = d.meeting_key
            AND l.session_key = d.session_key
        WHERE
        AND l.lap_duration IS NOT NULL
        AND l.lap_duration > 0
        AND COALESCE(l.is_outlier, false) = false
        ORDER BY l.lap_duration ASC
        LIMIT 1
        """

    def build_driver_performance_query(self, session_key, driver_name, metrics=["all"]):
        """Build query to get comprehensive driver performance data"""
        return """
        SELECT 
            d.full_name,
            d.team_name,
            s.session_name,
            COUNT(DISTINCT l.lap_number) AS total_laps,
            ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
            MIN(l.lap_duration) AS best_lap,
            MAX(l.lap_duration) AS worst_lap,
            ROUND(STDDEV(l.lap_duration)::numeric, 3) AS consistency,
            COUNT(CASE WHEN l.had_incident = true THEN 1 END) AS incidents,
            ROUND(AVG(l.duration_sector_1)::numeric, 3) AS avg_sector_1,
            ROUND(AVG(l.duration_sector_2)::numeric, 3) AS avg_sector_2,
            ROUND(AVG(l.duration_sector_3)::numeric, 3) AS avg_sector_3
        FROM laps_transformed l
        JOIN drivers_transformed d 
            ON l.driver_number = d.driver_number 
            AND l.meeting_key = d.meeting_key
            AND l.session_key = d.session_key
        JOIN sessions_transformed s 
            ON l.session_key = s.session_key 
            AND l.meeting_key = s.meeting_key
        WHERE l.session_key = :session_key
        AND UPPER(d.full_name) = UPPER(:driver_name)
        AND l.lap_duration IS NOT NULL
        AND l.lap_duration > 0
        AND COALESCE(l.is_outlier, false) = false
        GROUP BY d.full_name, d.team_name, s.session_name
        """

    def build_driver_comparison_query(self, session_key, driver1, driver2, comparison_metrics=["all"]):
        """Build query to compare performance between two drivers"""
        return """
        SELECT 
            d.full_name,
            d.team_name,
            COUNT(DISTINCT l.lap_number) AS total_laps,
            ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
            MIN(l.lap_duration) AS best_lap,
            MAX(l.lap_duration) AS worst_lap,
            ROUND(STDDEV(l.lap_duration)::numeric, 3) AS consistency,
            COUNT(CASE WHEN l.had_incident = true THEN 1 END) AS incidents
        FROM laps_transformed l
        JOIN drivers_transformed d 
            ON l.driver_number = d.driver_number 
            AND l.meeting_key = d.meeting_key
            AND l.session_key = d.session_key
        WHERE l.session_key = :session_key
        AND UPPER(d.full_name) IN (UPPER(:driver1), UPPER(:driver2))
        AND l.lap_duration IS NOT NULL
        AND l.lap_duration > 0
        AND COALESCE(l.is_outlier, false) = false
        GROUP BY d.full_name, d.team_name
        ORDER BY avg_lap ASC
        """

    def build_race_results_query(self, session_key, result_type="full_results", include_lap_times=False):
        """Build query to get race results with positions and details"""
        limit_clause = ""
        if result_type == "top_10":
            limit_clause = "LIMIT 10"
        elif result_type == "podium":
            limit_clause = "LIMIT 3"
        elif result_type == "winner_only":
            limit_clause = "LIMIT 1"

        if include_lap_times:
            return """
            WITH final_positions AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY driver_number ORDER BY date DESC) AS rn
                FROM positions_transformed
                WHERE session_key = :session_key
            ),
            driver_info AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY driver_number, meeting_key ORDER BY id) AS rn
                FROM drivers_transformed
                WHERE meeting_key = (SELECT meeting_key FROM sessions_transformed WHERE session_key = :session_key)
            )
            SELECT 
                d.full_name,
                d.team_name,
                p.position AS finish_position,
                MIN(l.lap_duration) AS best_lap,
                ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
                COUNT(l.lap_number) AS total_laps
            FROM final_positions p
            JOIN driver_info d 
                ON p.driver_number = d.driver_number AND p.meeting_key = d.meeting_key
            LEFT JOIN laps_transformed l ON p.driver_number = l.driver_number 
                AND p.meeting_key = l.meeting_key 
                AND p.session_key = l.session_key
            WHERE p.rn = 1
            AND d.rn = 1
            AND p.position IS NOT NULL
            GROUP BY d.full_name, d.team_name, p.position
            ORDER BY p.position ASC
            """ + limit_clause
        else:
            return """
            WITH final_positions AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY driver_number ORDER BY date DESC) AS rn
                FROM positions_transformed
                WHERE session_key = :session_key
            ),
            driver_info AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY driver_number, meeting_key ORDER BY id) AS rn
                FROM drivers_transformed
                WHERE meeting_key = (SELECT meeting_key FROM sessions_transformed WHERE session_key = :session_key)
            )
            SELECT 
                d.full_name,
                d.team_name,
                p.position AS finish_position
            FROM final_positions p
            JOIN driver_info d 
                ON p.driver_number = d.driver_number AND p.meeting_key = d.meeting_key
            WHERE p.rn = 1
            AND d.rn = 1
            AND p.position IS NOT NULL
            ORDER BY p.position ASC
            """ + limit_clause

    def build_pit_stop_analysis_query(self, session_key, driver_filter=None, team_filter=None, analysis_type="all"):
        """Build query to analyze pit stop strategy and performance"""
        where_conditions = ["ps.session_key = :session_key"]
        params = {"session_key": session_key}
        
        if driver_filter:
            where_conditions.append("UPPER(d.full_name) = UPPER(:driver_filter)")
            params["driver_filter"] = driver_filter
        if team_filter:
            where_conditions.append("UPPER(d.team_name) = UPPER(:team_filter)")
            params["team_filter"] = team_filter
        
        where_clause = " AND ".join(where_conditions)

        return """
        SELECT 
            d.full_name,
            d.team_name,
            COUNT(*) AS total_stops,
            ROUND(AVG(ps.pit_duration)::numeric, 2) AS avg_pit_time,
            MIN(ps.pit_duration) AS fastest_stop,
            MAX(ps.pit_duration) AS slowest_stop,
            MIN(ps.lap_number) AS first_stop,
            MAX(ps.lap_number) AS last_stop,
            COUNT(CASE WHEN ps.long_pit_stop = true THEN 1 END) AS long_stops
        FROM pit_stops_transformed ps
        JOIN drivers_transformed d ON ps.driver_number = d.driver_number 
            AND ps.meeting_key = d.meeting_key
            AND ps.session_key = d.session_key
        WHERE """ + where_clause + """
        GROUP BY d.full_name, d.team_name
        ORDER BY avg_pit_time ASC
        """

    def build_incident_investigation_query(self, session_key, driver_name, lap_number=None, context_laps=3, investigation_type="all"):
        """Build query to investigate incidents or unusual performance patterns"""
        lap_start = lap_number - context_laps if lap_number else 1
        lap_end = lap_number + context_laps if lap_number else 999

        return """
        SELECT 
            l.lap_number,
            l.lap_duration,
            l.duration_sector_1,
            l.duration_sector_2,
            l.duration_sector_3,
            l.had_incident,
            l.safety_car_lap,
            l.is_outlier,
            d.full_name,
            d.team_name
        FROM laps_transformed l
        JOIN drivers_transformed d ON l.driver_number = d.driver_number 
            AND l.meeting_key = d.meeting_key
            AND l.session_key = d.session_key
        WHERE l.session_key = :session_key
        AND UPPER(d.full_name) = UPPER(:driver_name)
        AND l.lap_number BETWEEN :lap_start AND :lap_end
        ORDER BY l.lap_number
        """

    def build_explore_schema_query(self, table_name=None, detail_level="overview"):
        """Build query to explore database schema (metadata query)"""
        if table_name:
            return """
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = LOWER(:table_name)
            ORDER BY ordinal_position
            """
        else:
            return """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """

    def build_session_info_query(self, session_key):
        """Build query to get basic session information"""
        return """
        SELECT 
            s.session_name,
            s.session_type,
            s.date_start,
            s.date_end,
            m.meeting_name,
            m.country_name
        FROM sessions_transformed s
        JOIN meetings m ON s.meeting_key = m.meeting_key
        WHERE s.session_key = :session_key
        """