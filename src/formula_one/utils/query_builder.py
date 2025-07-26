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
        
        # Handle Qualifying case with special logic
        if session_type.lower() in ["qualifying", "qual"]:
            return """
            SELECT session_key, session_name, session_type, date_start, date_end
            FROM sessions_transformed 
            WHERE meeting_key = :meeting_key 
            AND session_name LIKE :session_type
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
        SELECT *
        FROM (
            SELECT DISTINCT ON (l.driver_number)
                l.driver_number,
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
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number
            WHERE {where_clause}
                AND l.lap_duration IS NOT NULL
                AND l.lap_duration > 0
                AND COALESCE(l.is_outlier, false) = false
            ORDER BY l.driver_number, l.lap_duration ASC
        ) fastest_laps
        ORDER BY lap_duration ASC
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

    def build_driver_comparison_query(self, session_key, drivers, comparison_metrics=["all"]):
        """Build query to compare performance between multiple drivers"""
        
        # Build dynamic driver filter
        driver_filters = []
        params = {"session_key": session_key}
        
        for i, driver in enumerate(drivers):
            param_name = f"driver{i+1}"
            driver_filters.append(f"UPPER(d.full_name) = UPPER(:{param_name})")
            params[param_name] = driver
        
        driver_filter_clause = " OR ".join(driver_filters)
        
        return f"""
        SELECT 
            d.full_name,
            d.team_name,
            COUNT(DISTINCT l.lap_number) AS total_laps,
            ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
            MIN(l.lap_duration) AS best_lap,
            MAX(l.lap_duration) AS worst_lap,
            ROUND(STDDEV(l.lap_duration)::numeric, 3) AS consistency,
            COUNT(CASE WHEN l.had_incident = true THEN 1 END) AS incidents
        FROM drivers_transformed d
        LEFT JOIN laps_transformed l 
            ON d.driver_number = l.driver_number 
            AND d.meeting_key = l.meeting_key 
            AND d.session_key = l.session_key
            AND l.lap_duration IS NOT NULL
            AND l.lap_duration > 0
            AND COALESCE(l.is_outlier, false) = false
        WHERE d.session_key = :session_key
        AND ({driver_filter_clause})
        GROUP BY d.full_name, d.team_name
        ORDER BY best_lap ASC
        """

    def build_race_results_query(self, session_key, result_type="full_results", include_lap_times=False):
        """Build query to get race results with positions and details"""
        limit_clause = ""
        if result_type == "winner_only":
            limit_clause = "LIMIT 1"
        elif result_type == "podium":
            limit_clause = "LIMIT 3"
        elif result_type.startswith("top_"):
            try:
                num = int(result_type.split("_")[1])
                limit_clause = f"LIMIT {num}"
            except:
                limit_clause = "LIMIT 10"  # Default
        else:
            limit_clause = "LIMIT 10"  # Default

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

        if analysis_type == "summary":
            # ✅ Summary statistics with correct schema
            return """
            SELECT 
                d.full_name,
                d.team_name,
                COUNT(*) AS total_stops,
                ROUND(AVG(ps.pit_duration)::numeric, 6) AS avg_pit_time,
                MIN(ps.pit_duration) AS fastest_stop,
                MAX(ps.pit_duration) AS slowest_stop,
                MIN(ps.lap_number) AS first_stop,
                MAX(ps.lap_number) AS last_stop,
                COUNT(CASE WHEN ps.long_pit_stop = true THEN 1 END) AS long_stops,
                COUNT(CASE WHEN ps.normal_pit_stop = true THEN 1 END) AS normal_stops,
                COUNT(CASE WHEN ps.penalty_pit_stop = true THEN 1 END) AS penalty_stops
            FROM pit_stops_transformed ps
            JOIN drivers_transformed d ON ps.driver_number = d.driver_number 
                AND ps.meeting_key = d.meeting_key
                AND ps.session_key = d.session_key
            WHERE """ + where_clause + """
            GROUP BY d.full_name, d.team_name
            ORDER BY avg_pit_time ASC
            """
        else:
            # ✅ Individual pit stop details with correct schema
            return """
            SELECT 
                d.full_name,
                d.team_name,
                ps.lap_number,
                ps.pit_duration,                    -- Full precision, no rounding
                ps.normal_pit_stop,
                ps.long_pit_stop,
                ps.penalty_pit_stop,
                ps.is_outlier,
                ps.created_at
            FROM pit_stops_transformed ps
            JOIN drivers_transformed d ON ps.driver_number = d.driver_number 
                AND ps.meeting_key = d.meeting_key
                AND ps.session_key = d.session_key
            WHERE """ + where_clause + """
            ORDER BY ps.pit_duration ASC, d.full_name, ps.lap_number
            """
    
    def build_tire_strategy_query(self, session_key, driver_filter=None, team_filter=None, strategy_type="all"):
        """Build query to analyze tire strategy and stint information"""
        
        where_conditions = ["s.session_key = :session_key"]
        params = {"session_key": session_key}
        
        if driver_filter:
            where_conditions.append("UPPER(d.full_name) = UPPER(:driver_filter)")
            params["driver_filter"] = driver_filter
        if team_filter:
            where_conditions.append("d.team_name ILIKE :team_filter_pattern")
            params["team_filter_pattern"] = f"%{team_filter}%"
        
        where_clause = " AND ".join(where_conditions)
        
        return f"""
        SELECT 
            d.full_name,
            d.team_name,
            ROW_NUMBER() OVER (PARTITION BY d.driver_number ORDER BY s.lap_start) AS stint_number,
            s.lap_start AS start_lap,
            s.lap_end AS end_lap,
            s.compound AS tire_compound,
            (s.lap_end - s.lap_start + 1) AS stint_length,
            ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap_time
        FROM stints_transformed s
        JOIN drivers_transformed d 
            ON s.driver_number = d.driver_number 
            AND s.meeting_key = d.meeting_key 
            AND s.session_key = d.session_key
        LEFT JOIN laps_transformed l 
            ON s.driver_number = l.driver_number 
            AND s.meeting_key = l.meeting_key 
            AND s.session_key = l.session_key
            AND l.lap_number BETWEEN s.lap_start AND s.lap_end
            AND l.lap_duration IS NOT NULL
            AND l.lap_duration > 0
            AND COALESCE(l.is_outlier, false) = false
        WHERE {where_clause}
        GROUP BY d.driver_number, d.full_name, d.team_name, s.lap_start, s.lap_end, s.compound
        ORDER BY d.full_name, s.lap_start
        """

    
    def build_qualifying_results_query(self, session_key, result_type="full_results"):
        """Build query to get qualifying results with positions and lap times"""
        
        base_query = f"""
        WITH best_laps AS (
            SELECT driver_number, lap_duration AS best_lap
            FROM (
                SELECT 
                    driver_number,
                    lap_duration,
                    ROW_NUMBER() OVER (PARTITION BY driver_number ORDER BY lap_duration ASC) AS rn
                FROM laps_transformed
                WHERE session_key = :session_key
                AND lap_duration IS NOT NULL 
                AND lap_duration > 0
                AND COALESCE(is_outlier, false) = false
            ) sub
            WHERE rn = 1
        ),
        q1_laps AS (
            SELECT driver_number, lap_duration AS q1_time
            FROM (
                SELECT 
                    driver_number,
                    lap_duration,
                    ROW_NUMBER() OVER (PARTITION BY driver_number ORDER BY lap_duration ASC) AS rn
                FROM laps_transformed
                WHERE session_key = :session_key
                AND lap_number <= 5
                AND lap_duration IS NOT NULL 
                AND lap_duration > 0
                AND COALESCE(is_outlier, false) = false
            ) sub
            WHERE rn = 1
        ),
        q2_laps AS (
            SELECT driver_number, lap_duration AS q2_time
            FROM (
                SELECT 
                    driver_number,
                    lap_duration,
                    ROW_NUMBER() OVER (PARTITION BY driver_number ORDER BY lap_duration ASC) AS rn
                FROM laps_transformed
                WHERE session_key = :session_key
                AND lap_number > 5 AND lap_number <= 10
                AND lap_duration IS NOT NULL 
                AND lap_duration > 0
                AND COALESCE(is_outlier, false) = false
            ) sub
            WHERE rn = 1
        ),
        q3_laps AS (
            SELECT driver_number, lap_duration AS q3_time
            FROM (
                SELECT 
                    driver_number,
                    lap_duration,
                    ROW_NUMBER() OVER (PARTITION BY driver_number ORDER BY lap_duration ASC) AS rn
                FROM laps_transformed
                WHERE session_key = :session_key
                AND lap_number > 10
                AND lap_duration IS NOT NULL 
                AND lap_duration > 0
                AND COALESCE(is_outlier, false) = false
            ) sub
            WHERE rn = 1
        ),
        qualified AS (
            SELECT 
                d.driver_number,
                d.full_name,
                d.team_name,
                bl.best_lap,
                q1.q1_time,
                q2.q2_time,
                q3.q3_time,
                ROW_NUMBER() OVER (ORDER BY bl.best_lap ASC) AS qualifying_position
            FROM drivers_transformed d
            JOIN best_laps bl ON d.driver_number = bl.driver_number
            LEFT JOIN q1_laps q1 ON d.driver_number = q1.driver_number
            LEFT JOIN q2_laps q2 ON d.driver_number = q2.driver_number
            LEFT JOIN q3_laps q3 ON d.driver_number = q3.driver_number
            WHERE d.session_key = :session_key
        )
        SELECT 
            full_name,
            team_name,
            qualifying_position,
            best_lap,
            q1_time,
            CASE WHEN qualifying_position <= 15 THEN q2_time ELSE NULL END AS q2_time,
            CASE WHEN qualifying_position <= 10 THEN q3_time ELSE NULL END AS q3_time
        FROM qualified
        """
        
        # Add result type filtering
        if result_type == "top_10":
            base_query += " WHERE qualifying_position <= 10"
        elif result_type == "podium":
            base_query += " WHERE qualifying_position <= 3"
        elif result_type == "q3":
            base_query += " WHERE qualifying_position <= 10"
        elif result_type == "q2":
            base_query += " WHERE qualifying_position <= 15"
        elif result_type == "q1":
            # No filter needed - all drivers who set times
            pass
        
        base_query += " ORDER BY qualifying_position ASC"
        
        return base_query
    

    def build_team_performance_query(self, session_key, team_name, metrics=["all"]):
        """Build query to get performance data for all drivers in a team"""
        
        return f"""
        WITH team_summary AS (
            SELECT 
                d.team_name,
                COUNT(DISTINCT l.driver_number || '-' || l.lap_number) AS total_laps,
                ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
                MIN(l.lap_duration) AS best_lap,
                ROUND(STDDEV(l.lap_duration)::numeric, 3) AS consistency
            FROM drivers_transformed d
            LEFT JOIN laps_transformed l 
                ON d.driver_number = l.driver_number 
                AND d.meeting_key = l.meeting_key 
                AND d.session_key = l.session_key
                AND l.lap_duration IS NOT NULL
                AND l.lap_duration > 0
                AND COALESCE(l.is_outlier, false) = false
            WHERE d.session_key = :session_key
            AND d.team_name ILIKE :team_name_pattern
            GROUP BY d.team_name
        ),
        driver_details AS (
            SELECT 
                d.team_name,
                d.full_name,
                p_final.position AS session_position,
                COUNT(DISTINCT l.lap_number) AS driver_laps,
                ROUND(AVG(l.lap_duration)::numeric, 3) AS driver_avg_lap,
                MIN(l.lap_duration) AS driver_best_lap
            FROM drivers_transformed d
            LEFT JOIN laps_transformed l 
                ON d.driver_number = l.driver_number 
                AND d.meeting_key = l.meeting_key 
                AND d.session_key = l.session_key
                AND l.lap_duration IS NOT NULL
                AND l.lap_duration > 0
                AND COALESCE(l.is_outlier, false) = false
            LEFT JOIN (
                SELECT DISTINCT ON (driver_number)
                    driver_number,
                    position
                FROM positions_transformed
                WHERE session_key = :session_key
                ORDER BY driver_number, date DESC
            ) p_final ON d.driver_number = p_final.driver_number
            WHERE d.session_key = :session_key
            AND d.team_name ILIKE :team_name_pattern
            GROUP BY d.team_name, d.full_name, p_final.position
        )
        SELECT 
            ts.team_name,
            ts.total_laps,
            ts.avg_lap,
            ts.best_lap,
            ts.consistency,
            ARRAY_AGG(dd.session_position ORDER BY dd.session_position) AS driver_positions,
            ARRAY_AGG(dd.full_name ORDER BY dd.session_position) AS driver_names
        FROM team_summary ts
        JOIN driver_details dd ON ts.team_name = dd.team_name
        GROUP BY ts.team_name, ts.total_laps, ts.avg_lap, ts.best_lap, ts.consistency
        ORDER BY ts.avg_lap ASC
        """
    
    def build_team_comparison_query(self, session_key, teams, comparison_metrics=["all"]):
        """Build query to compare performance between multiple teams with driver breakdown"""
        
        # Build dynamic team filter
        team_filters = []
        params = {"session_key": session_key}
        
        for i, team in enumerate(teams):
            param_name = f"team{i+1}_pattern"
            team_filters.append(f"d.team_name ILIKE :{param_name}")
            params[param_name] = f"%{team}%"
        
        team_filter_clause = " OR ".join(team_filters)
        
        return f"""
        WITH final_positions AS (
            SELECT DISTINCT ON (driver_number)
                driver_number,
                position
            FROM positions_transformed
            WHERE session_key = :session_key
            ORDER BY driver_number, date DESC
        ),
        team_stats AS (
            SELECT 
                'team' AS level,
                d.team_name,
                NULL AS full_name,
                COUNT(DISTINCT l.driver_number || '-' || l.lap_number) AS total_laps,
                ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
                MIN(l.lap_duration) AS best_lap,
                ROUND(STDDEV(l.lap_duration)::numeric, 3) AS consistency,
                MIN(p.position) AS best_position,
                ROUND(AVG(p.position)::numeric, 1) AS avg_position
            FROM drivers_transformed d
            LEFT JOIN laps_transformed l 
                ON d.driver_number = l.driver_number 
                AND d.meeting_key = l.meeting_key 
                AND d.session_key = l.session_key
                AND l.lap_duration IS NOT NULL
                AND l.lap_duration > 0
                AND COALESCE(l.is_outlier, false) = false
            LEFT JOIN final_positions p ON d.driver_number = p.driver_number
            WHERE d.session_key = :session_key
            AND ({team_filter_clause})
            GROUP BY d.team_name
        ),
        driver_stats AS (
            SELECT 
                'driver' AS level,
                d.team_name,
                d.full_name,
                COUNT(DISTINCT l.lap_number) AS total_laps,
                ROUND(AVG(l.lap_duration)::numeric, 3) AS avg_lap,
                MIN(l.lap_duration) AS best_lap,
                ROUND(STDDEV(l.lap_duration)::numeric, 3) AS consistency,
                p.position AS best_position,
                NULL::numeric AS avg_position
            FROM drivers_transformed d
            LEFT JOIN laps_transformed l 
                ON d.driver_number = l.driver_number 
                AND d.meeting_key = l.meeting_key 
                AND d.session_key = l.session_key
                AND l.lap_duration IS NOT NULL
                AND l.lap_duration > 0
                AND COALESCE(l.is_outlier, false) = false
            LEFT JOIN final_positions p ON d.driver_number = p.driver_number
            WHERE d.session_key = :session_key
            AND ({team_filter_clause})
            GROUP BY d.team_name, d.full_name, p.position
        )
        SELECT *
        FROM (
            SELECT * FROM team_stats
            UNION ALL
            SELECT * FROM driver_stats
        ) all_data
        ORDER BY team_name, level, best_lap
        """
    
    def build_smart_incident_query(self, session_key, driver_name=None, lap_number=None, target_drivers=None, max_results=15):
        """
        Smart incident query that uses multiple passes to find the most relevant incidents
        
        Args:
            session_key: Session to analyze
            driver_name: Specific driver to focus on
            lap_number: Specific lap to focus on
            target_drivers: List of specific drivers to look for (e.g., ["Lando Norris", "Oscar Piastri"])
            max_results: Maximum results per pass
        """
        
        # Build base filters
        driver_filter = ""
        if driver_name:
            driver_filter = "AND UPPER(d.full_name) = UPPER(:driver_name)"
        
        lap_filter = ""
        if lap_number:
            lap_filter = "AND l.lap_number BETWEEN :lap_start AND :lap_end"
        
        target_driver_filter = ""
        if target_drivers:
            driver_conditions = []
            for i, driver in enumerate(target_drivers):
                driver_conditions.append(f"UPPER(d.full_name) = UPPER(:target_driver_{i})")
            target_driver_filter = f"AND ({' OR '.join(driver_conditions)})"
        
        return f"""
        WITH incident_ranked AS (
            SELECT 
                l.lap_number,
                d.full_name,
                d.team_name,
                l.lap_duration,
                l.had_incident,
                l.safety_car_lap,
                l.is_outlier,
                rc.flag,
                rc.category,
                rc.message,
                pos.position,
                l.created_at,
                
                -- Smart relevance scoring
                CASE 
                    -- Exact lap match gets highest priority
                    WHEN l.lap_number = :target_lap THEN 100
                    -- Target drivers get high priority
                    WHEN :target_drivers IS NOT NULL AND d.full_name = ANY(:target_drivers) THEN 90
                    -- Specific driver queries
                    WHEN :driver_name IS NOT NULL AND UPPER(d.full_name) = UPPER(:driver_name) THEN 80
                    -- Crashes and major incidents
                    WHEN rc.category = 'Crash' OR l.had_incident = true THEN 70
                    -- Red flags
                    WHEN rc.flag = 'Red' THEN 60
                    -- Yellow flags
                    WHEN rc.flag = 'Yellow' THEN 50
                    -- Safety car
                    WHEN l.safety_car_lap = true OR rc.category = 'Safety Car' THEN 40
                    -- Slow laps
                    WHEN l.is_outlier = true THEN 30
                    -- Messages containing incident keywords
                    WHEN rc.message IS NOT NULL AND (
                        LOWER(rc.message) LIKE '%incident%' OR
                        LOWER(rc.message) LIKE '%crash%' OR
                        LOWER(rc.message) LIKE '%collision%' OR
                        LOWER(rc.message) LIKE '%contact%' OR
                        LOWER(rc.message) LIKE '%spin%' OR
                        LOWER(rc.message) LIKE '%off%' OR
                        LOWER(rc.message) LIKE '%damage%'
                    ) THEN 25
                    ELSE 10
                END as relevance_score,
                
                -- Row number for limiting
                ROW_NUMBER() OVER (
                    ORDER BY 
                        CASE 
                            WHEN l.lap_number = :target_lap THEN 100
                            WHEN :target_drivers IS NOT NULL AND d.full_name = ANY(:target_drivers) THEN 90
                            WHEN :driver_name IS NOT NULL AND UPPER(d.full_name) = UPPER(:driver_name) THEN 80
                            WHEN rc.category = 'Crash' OR l.had_incident = true THEN 70
                            WHEN rc.flag = 'Red' THEN 60
                            WHEN rc.flag = 'Yellow' THEN 50
                            WHEN l.safety_car_lap = true OR rc.category = 'Safety Car' THEN 40
                            WHEN l.is_outlier = true THEN 30
                            WHEN rc.message IS NOT NULL AND (
                                LOWER(rc.message) LIKE '%incident%' OR
                                LOWER(rc.message) LIKE '%crash%' OR
                                LOWER(rc.message) LIKE '%collision%' OR
                                LOWER(rc.message) LIKE '%contact%' OR
                                LOWER(rc.message) LIKE '%spin%' OR
                                LOWER(rc.message) LIKE '%off%' OR
                                LOWER(rc.message) LIKE '%damage%'
                            ) THEN 25
                            ELSE 10
                        END DESC,
                        l.lap_number
                ) as global_rank

            FROM laps_transformed l
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number 
                AND l.meeting_key = d.meeting_key
                AND l.session_key = d.session_key
            LEFT JOIN race_control rc 
                ON rc.session_key = l.session_key
                AND rc.lap_number = l.lap_number
                AND (rc.driver_number = l.driver_number OR rc.driver_number IS NULL OR rc.driver_number = 0)
            LEFT JOIN LATERAL (
                SELECT p.position
                FROM positions_transformed p
                WHERE p.driver_number = l.driver_number
                AND p.session_key = l.session_key
                AND p.meeting_key = l.meeting_key
                AND p.date <= l.created_at
                ORDER BY p.date DESC
                LIMIT 1
            ) pos ON true
            WHERE l.session_key = :session_key
            {driver_filter}
            {lap_filter}
            {target_driver_filter}
            AND (
                rc.flag IS NOT NULL OR 
                l.had_incident = true OR 
                l.safety_car_lap = true OR 
                l.is_outlier = true OR
                (rc.message IS NOT NULL AND (
                    LOWER(rc.message) LIKE '%incident%' OR
                    LOWER(rc.message) LIKE '%crash%' OR
                    LOWER(rc.message) LIKE '%collision%' OR
                    LOWER(rc.message) LIKE '%contact%' OR
                    LOWER(rc.message) LIKE '%spin%' OR
                    LOWER(rc.message) LIKE '%off%' OR
                    LOWER(rc.message) LIKE '%damage%'
                ))
            )
        )
        SELECT 
            lap_number,
            full_name,
            team_name,
            lap_duration,
            had_incident,
            safety_car_lap,
            is_outlier,
            flag,
            category,
            message,
            position,
            created_at,
            relevance_score,
            global_rank
        FROM incident_ranked
        WHERE global_rank <= :max_results
        ORDER BY global_rank
        """

    def build_multi_pass_incident_query(self, session_key, driver_name=None, lap_number=None, target_drivers=None):
        """
        Multi-pass incident query that tries different strategies to find relevant incidents
        """
        
        # Build the query parts
        query_parts = []

        # Pass 1: High-priority incidents (crashes, major incidents)
        pass1_query = """
        SELECT 
            l.lap_number,
            d.full_name,
            d.team_name,
            l.lap_duration,
            l.had_incident,
            l.safety_car_lap,
            l.is_outlier,
            rc.flag,
            rc.category,
            rc.message,
            pos.position,
            l.created_at,
            'PASS1' as pass_type
        FROM laps_transformed l
        JOIN drivers_transformed d 
            ON l.driver_number = d.driver_number 
            AND l.meeting_key = d.meeting_key
            AND l.session_key = d.session_key
        LEFT JOIN race_control rc 
            ON rc.session_key = l.session_key
            AND rc.lap_number = l.lap_number
            AND (rc.driver_number = l.driver_number OR rc.driver_number IS NULL OR rc.driver_number = 0)
        LEFT JOIN LATERAL (
            SELECT p.position
            FROM positions_transformed p
            WHERE p.driver_number = l.driver_number
            AND p.session_key = l.session_key
            AND p.meeting_key = l.meeting_key
            AND p.date <= l.created_at
            ORDER BY p.date DESC
            LIMIT 1
        ) pos ON true
        WHERE l.session_key = :session_key
        AND (
            rc.category = 'Crash' OR 
            l.had_incident = true OR 
            rc.flag = 'Red' OR
            (rc.message IS NOT NULL AND (
                LOWER(rc.message) LIKE '%crash%' OR
                LOWER(rc.message) LIKE '%collision%' OR
                LOWER(rc.message) LIKE '%contact%'
            ))
        )
        """
        query_parts.append(pass1_query)

        # Pass 2: Target lap incidents
        if lap_number is not None:
            pass2_query = """
            UNION ALL
            SELECT 
                l.lap_number,
                d.full_name,
                d.team_name,
                l.lap_duration,
                l.had_incident,
                l.safety_car_lap,
                l.is_outlier,
                rc.flag,
                rc.category,
                rc.message,
                pos.position,
                l.created_at,
                'PASS2' as pass_type
            FROM laps_transformed l
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number 
                AND l.meeting_key = d.meeting_key
                AND l.session_key = d.session_key
            LEFT JOIN race_control rc 
                ON rc.session_key = l.session_key
                AND rc.lap_number = l.lap_number
                AND (rc.driver_number = l.driver_number OR rc.driver_number IS NULL OR rc.driver_number = 0)
            LEFT JOIN LATERAL (
                SELECT p.position
                FROM positions_transformed p
                WHERE p.driver_number = l.driver_number
                AND p.session_key = l.session_key
                AND p.meeting_key = l.meeting_key
                AND p.date <= l.created_at
                ORDER BY p.date DESC
                LIMIT 1
            ) pos ON true
            WHERE l.session_key = :session_key
            AND l.lap_number BETWEEN :lap_start AND :lap_end
            AND (
                rc.flag IS NOT NULL OR 
                l.had_incident = true OR 
                l.safety_car_lap = true OR 
                l.is_outlier = true
            )
            """
            query_parts.append(pass2_query)

        # Pass 3: Target driver incidents
        if target_drivers:
            driver_conditions = []
            for i, driver in enumerate(target_drivers):
                driver_conditions.append(f"UPPER(d.full_name) = UPPER(:target_driver_{i})")
            
            pass3_query = f"""
            UNION ALL
            SELECT 
                l.lap_number,
                d.full_name,
                d.team_name,
                l.lap_duration,
                l.had_incident,
                l.safety_car_lap,
                l.is_outlier,
                rc.flag,
                rc.category,
                rc.message,
                pos.position,
                l.created_at,
                'PASS3' as pass_type
            FROM laps_transformed l
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number 
                AND l.meeting_key = d.meeting_key
                AND l.session_key = d.session_key
            LEFT JOIN race_control rc 
                ON rc.session_key = l.session_key
                AND rc.lap_number = l.lap_number
                AND (rc.driver_number = l.driver_number OR rc.driver_number IS NULL OR rc.driver_number = 0)
            LEFT JOIN LATERAL (
                SELECT p.position
                FROM positions_transformed p
                WHERE p.driver_number = l.driver_number
                AND p.session_key = l.session_key
                AND p.meeting_key = l.meeting_key
                AND p.date <= l.created_at
                ORDER BY p.date DESC
                LIMIT 1
            ) pos ON true
            WHERE l.session_key = :session_key
            AND ({' OR '.join(driver_conditions)})
            AND (
                rc.flag IS NOT NULL OR 
                l.had_incident = true OR 
                l.safety_car_lap = true OR 
                l.is_outlier = true OR
                (rc.message IS NOT NULL AND (
                    LOWER(rc.message) LIKE '%incident%' OR
                    LOWER(rc.message) LIKE '%crash%' OR
                    LOWER(rc.message) LIKE '%collision%' OR
                    LOWER(rc.message) LIKE '%contact%' OR
                    LOWER(rc.message) LIKE '%spin%' OR
                    LOWER(rc.message) LIKE '%off%' OR
                    LOWER(rc.message) LIKE '%damage%'
                ))
            )
            """
            query_parts.append(pass3_query)

        # Final full query with ORDER BY outside the UNION
        full_query = f"""
        SELECT * FROM (
            {" ".join(query_parts)}
        ) AS unioned_results
        ORDER BY 
            CASE pass_type 
                WHEN 'PASS1' THEN 1 
                WHEN 'PASS2' THEN 2 
                WHEN 'PASS3' THEN 3 
                ELSE 4
            END,
            lap_number
        LIMIT 30
        """

        return full_query
    
    def build_sector_analysis_query(self, session_key, driver_filter=None, team_filter=None, sector_analysis_type="all"):
        """
        Build query to analyze sector times and identify strengths/weaknesses
        
        Args:
            session_key: Session to analyze
            driver_filter: Specific driver to focus on
            team_filter: Specific team to focus on
            sector_analysis_type: "all", "best_sectors", "consistency", "comparison"
        """
        
        # Build filter conditions
        where_conditions = ["l.session_key = :session_key"]
        params = {"session_key": session_key}
        
        if driver_filter:
            where_conditions.append("UPPER(d.full_name) = UPPER(:driver_filter)")
            params["driver_filter"] = driver_filter
        
        if team_filter:
            where_conditions.append("UPPER(d.team_name) = UPPER(:team_filter)")
            params["team_filter"] = team_filter
        
        where_clause = " AND ".join(where_conditions)
        
        # Different query types based on analysis type
        if sector_analysis_type == "best_sectors":
            return f"""
            SELECT 
                d.full_name,
                d.team_name,
                MIN(l.duration_sector_1) as best_sector1,
                MIN(l.duration_sector_2) as best_sector2,
                MIN(l.duration_sector_3) as best_sector3,
                ROUND(AVG(l.duration_sector_1)::numeric, 3) as avg_sector1,
                ROUND(AVG(l.duration_sector_2)::numeric, 3) as avg_sector2,
                ROUND(AVG(l.duration_sector_3)::numeric, 3) as avg_sector3,
                COUNT(DISTINCT l.lap_number) as total_laps,
                -- Sector consistency (lower is better)
                ROUND(STDDEV(l.duration_sector_1)::numeric, 3) as sector1_consistency,
                ROUND(STDDEV(l.duration_sector_2)::numeric, 3) as sector2_consistency,
                ROUND(STDDEV(l.duration_sector_3)::numeric, 3) as sector3_consistency
            FROM laps_transformed l
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number 
                AND l.meeting_key = d.meeting_key
                AND l.session_key = d.session_key
            WHERE {where_clause}
            AND l.duration_sector_1 IS NOT NULL
            AND l.duration_sector_2 IS NOT NULL
            AND l.duration_sector_3 IS NOT NULL
            AND l.duration_sector_1 > 0
            AND l.duration_sector_2 > 0
            AND l.duration_sector_3 > 0
            AND COALESCE(l.is_outlier, false) = false
            GROUP BY d.full_name, d.team_name
            ORDER BY (MIN(l.duration_sector_1) + MIN(l.duration_sector_2) + MIN(l.duration_sector_3))
            LIMIT 20
            """
        
        elif sector_analysis_type == "consistency":
            return f"""
            SELECT 
                d.full_name,
                d.team_name,
                ROUND(AVG(l.duration_sector_1)::numeric, 3) as avg_sector1,
                ROUND(AVG(l.duration_sector_2)::numeric, 3) as avg_sector2,
                ROUND(AVG(l.duration_sector_3)::numeric, 3) as avg_sector3,
                ROUND(STDDEV(l.duration_sector_1)::numeric, 3) as sector1_consistency,
                ROUND(STDDEV(l.duration_sector_2)::numeric, 3) as sector2_consistency,
                ROUND(STDDEV(l.duration_sector_3)::numeric, 3) as sector3_consistency,
                COUNT(DISTINCT l.lap_number) as total_laps,
                -- Overall consistency score (lower is better)
                ROUND((STDDEV(l.duration_sector_1) + STDDEV(l.duration_sector_2) + STDDEV(l.duration_sector_3))::numeric, 3) as overall_consistency
            FROM laps_transformed l
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number 
                AND l.meeting_key = d.meeting_key
                AND l.session_key = d.session_key
            WHERE {where_clause}
            AND l.duration_sector_1 IS NOT NULL
            AND l.duration_sector_2 IS NOT NULL
            AND l.duration_sector_3 IS NOT NULL
            AND l.duration_sector_1 > 0
            AND l.duration_sector_2 > 0
            AND l.duration_sector_3 > 0
            AND COALESCE(l.is_outlier, false) = false
            GROUP BY d.full_name, d.team_name
            HAVING COUNT(DISTINCT l.lap_number) >= 5  -- Only drivers with sufficient laps
            ORDER BY overall_consistency
            LIMIT 20
            """
        
        elif sector_analysis_type == "comparison":
            return f"""
            WITH sector_stats AS (
                SELECT 
                    d.full_name,
                    d.team_name,
                    MIN(l.duration_sector_1) as best_sector1,
                    MIN(l.duration_sector_2) as best_sector2,
                    MIN(l.duration_sector_3) as best_sector3,
                    ROUND(AVG(l.duration_sector_1)::numeric, 3) as avg_sector1,
                    ROUND(AVG(l.duration_sector_2)::numeric, 3) as avg_sector2,
                    ROUND(AVG(l.duration_sector_3)::numeric, 3) as avg_sector3,
                    COUNT(DISTINCT l.lap_number) as total_laps
                FROM laps_transformed l
                JOIN drivers_transformed d 
                    ON l.driver_number = d.driver_number 
                    AND l.meeting_key = d.meeting_key
                    AND l.session_key = d.session_key
                WHERE {where_clause}
                AND l.duration_sector_1 IS NOT NULL
                AND l.duration_sector_2 IS NOT NULL
                AND l.duration_sector_3 IS NOT NULL
                AND l.duration_sector_1 > 0
                AND l.duration_sector_2 > 0
                AND l.duration_sector_3 > 0
                AND COALESCE(l.is_outlier, false) = false
                GROUP BY d.full_name, d.team_name
            ),
            session_bests AS (
                SELECT 
                    MIN(best_sector1) as session_best_sector1,
                    MIN(best_sector2) as session_best_sector2,
                    MIN(best_sector3) as session_best_sector3
                FROM sector_stats
            )
            SELECT 
                s.full_name,
                s.team_name,
                s.best_sector1,
                s.best_sector2,
                s.best_sector3,
                s.avg_sector1,
                s.avg_sector2,
                s.avg_sector3,
                -- Gap to session best in each sector
                ROUND((s.best_sector1 - sb.session_best_sector1)::numeric, 3) as sector1_gap_to_best,
                ROUND((s.best_sector2 - sb.session_best_sector2)::numeric, 3) as sector2_gap_to_best,
                ROUND((s.best_sector3 - sb.session_best_sector3)::numeric, 3) as sector3_gap_to_best,
                s.total_laps
            FROM sector_stats s
            CROSS JOIN session_bests sb
            ORDER BY (s.best_sector1 + s.best_sector2 + s.best_sector3)
            LIMIT 20
            """
        
        else:  # "all" - comprehensive sector analysis
            return f"""
            SELECT 
                d.full_name,
                d.team_name,
                MIN(l.duration_sector_1) as best_sector1,
                MIN(l.duration_sector_2) as best_sector2,
                MIN(l.duration_sector_3) as best_sector3,
                ROUND(AVG(l.duration_sector_1)::numeric, 3) as avg_sector1,
                ROUND(AVG(l.duration_sector_2)::numeric, 3) as avg_sector2,
                ROUND(AVG(l.duration_sector_3)::numeric, 3) as avg_sector3,
                COUNT(DISTINCT l.lap_number) as total_laps,
                -- Sector consistency
                ROUND(STDDEV(l.duration_sector_1)::numeric, 3) as sector1_consistency,
                ROUND(STDDEV(l.duration_sector_2)::numeric, 3) as sector2_consistency,
                ROUND(STDDEV(l.duration_sector_3)::numeric, 3) as sector3_consistency,
                -- Sector strengths/weaknesses
                CASE 
                    WHEN MIN(l.duration_sector_1) <= MIN(l.duration_sector_2) AND MIN(l.duration_sector_1) <= MIN(l.duration_sector_3) THEN 'Sector 1'
                    WHEN MIN(l.duration_sector_2) <= MIN(l.duration_sector_1) AND MIN(l.duration_sector_2) <= MIN(l.duration_sector_3) THEN 'Sector 2'
                    ELSE 'Sector 3'
                END as strongest_sector,
                CASE 
                    WHEN MIN(l.duration_sector_1) >= MIN(l.duration_sector_2) AND MIN(l.duration_sector_1) >= MIN(l.duration_sector_3) THEN 'Sector 1'
                    WHEN MIN(l.duration_sector_2) >= MIN(l.duration_sector_1) AND MIN(l.duration_sector_2) >= MIN(l.duration_sector_3) THEN 'Sector 2'
                    ELSE 'Sector 3'
                END as weakest_sector
            FROM laps_transformed l
            JOIN drivers_transformed d 
                ON l.driver_number = d.driver_number 
                AND l.meeting_key = d.meeting_key
                AND l.session_key = d.session_key
            WHERE {where_clause}
            AND l.duration_sector_1 IS NOT NULL
            AND l.duration_sector_2 IS NOT NULL
            AND l.duration_sector_3 IS NOT NULL
            AND l.duration_sector_1 > 0
            AND l.duration_sector_2 > 0
            AND l.duration_sector_3 > 0
            AND COALESCE(l.is_outlier, false) = false
            GROUP BY d.full_name, d.team_name
            HAVING COUNT(DISTINCT l.lap_number) >= 3  -- Only drivers with sufficient laps
            ORDER BY (MIN(l.duration_sector_1) + MIN(l.duration_sector_2) + MIN(l.duration_sector_3))
            LIMIT 20
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