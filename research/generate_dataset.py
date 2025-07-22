import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import re
from collections import defaultdict

class F1DatasetGenerator:
    def __init__(self):
        # Teams and drivers
        self.teams_drivers = {
            "McLaren": ["Lando NORRIS", "Oscar PIASTRI"],
            "Ferrari": ["Charles LECLERC", "Lewis HAMILTON"],
            "Red Bull Racing": ["Max VERSTAPPEN", "Yuki TSUNODA"],
            "Mercedes": ["George RUSSELL", "Andrea Kimi ANTONELLI"],
            "Aston Martin": ["Fernando ALONSO", "Lance STROLL"],
            "Alpine": ["Pierre GASLY", "Franco COLAPINTO"],
            "Haas": ["Oliver BEARMAN", "Esteban OCON"],
            "Racing Bulls": ["Liam LAWSON", "Isack HADJAR"],
            "Sauber": ["Nico HULKENBERG", "Gabriel BORTOLETO"],
            "Williams": ["Alex ALBON", "Carlos SAINZ"]
        }
        
        # Races
        self.races = {
            'australian': 'Australian Grand Prix',
            'chinese': 'Chinese Grand Prix',
            'japanese': 'Japanese Grand Prix',
            'bahrain': 'Bahrain Grand Prix',
            'saudi': 'Saudi Arabian Grand Prix',
            'miami': 'Miami Grand Prix',
            'emilia-romagna': 'Emilia‑Romagna Grand Prix',
            'monaco': 'Monaco Grand Prix',
            'spanish': 'Spanish Grand Prix',
            'canadian': 'Canadian Grand Prix',
            'austrian': 'Austrian Grand Prix',
            'british': 'British Grand Prix'
        }
        
        # Question templates
        self.question_templates = {
            "race_results": [
                "Who won the {race_name}?",
                "What position did {driver_name} finish in the {race_name}?",
                "Who came in {position} place at the {race_name}?",
                "What was the podium for the {race_name}?",
                "Who won the last race?",
                "What position did {driver_name} get in the most recent race?",
                "Who finished {position} in the {race_name}?",
                "What was {driver_name}'s result in the {race_name}?",
                "Who won the {race_name} this year?",
                "What position did {team_name} drivers finish in the {race_name}?",
                "Who came {position} in the {race_name}?",
                "What was {driver_name}'s finishing position in the {race_name}?",
                "Who took the checkered flag at the {race_name}?",
                "What position did {driver_name} end up in the {race_name}?",
                "Who was the winner of the {race_name}?"
            ],
            
            "qualifying_results": [
                "Who got pole position for the {race_name}?",
                "What was {driver_name}'s qualifying position for the {race_name}?",
                "Who qualified {position} for the {race_name}?",
                "What was the qualifying order for the {race_name}?",
                "Who got the fastest lap in qualifying for the {race_name}?",
                "What was {team_name}'s qualifying performance at the {race_name}?",
                "Who secured pole for the {race_name}?",
                "What was {driver_name}'s grid position for the {race_name}?",
                "Who qualified on pole for the {race_name}?",
                "What was the qualifying result for the {race_name}?",
                "Who got the fastest qualifying lap at the {race_name}?",
                "What position did {driver_name} qualify for the {race_name}?",
                "Who was fastest in qualifying for the {race_name}?",
                "What was {team_name}'s qualifying result at the {race_name}?",
                "Who took pole position at the {race_name}?"
            ],
            
            "driver_performance": [
                "How did {driver_name} perform this season?",
                "What was {driver_name}'s average position this year?",
                "How many points did {driver_name} score in the {race_name}?",
                "What was {driver_name}'s fastest lap in the {race_name}?",
                "How did {driver_name} perform in qualifying vs race?",
                "What was {driver_name}'s best result this season?",
                "How has {driver_name} been performing lately?",
                "What was {driver_name}'s performance in the {race_name}?",
                "How did {driver_name} do this year?",
                "What was {driver_name}'s strongest race this season?",
                "How consistent has {driver_name} been this year?",
                "What was {driver_name}'s worst result this season?",
                "How did {driver_name} perform in qualifying?",
                "What was {driver_name}'s race pace like?",
                "How did {driver_name} handle the {race_name}?"
            ],
            
            "team_performance": [
                "How did {team_name} perform this season?",
                "What was {team_name}'s best result this year?",
                "How many points did {team_name} score?",
                "What was {team_name}'s average qualifying position?",
                "How did {team_name} perform in the {race_name}?",
                "Which {team_name} driver performed better?",
                "What was {team_name}'s strongest race?",
                "How consistent was {team_name} this season?",
                "What was {team_name}'s qualifying performance?",
                "How did {team_name} compare to other teams?",
                "What was {team_name}'s race pace like?",
                "How did {team_name} handle different tracks?",
                "What was {team_name}'s strategy performance?",
                "How did {team_name} perform in wet conditions?",
                "What was {team_name}'s reliability like?"
            ],
            
            "fastest_laps": [
                "What was the fastest lap in the {race_name}?",
                "Who set the fastest lap at the {race_name}?",
                "What was {driver_name}'s fastest lap in the {race_name}?",
                "What was the fastest lap time at the {race_name}?",
                "Who got the fastest lap in the {race_name}?",
                "What was the quickest lap in the {race_name}?",
                "Who set the fastest lap time at the {race_name}?",
                "What was the fastest lap of the {race_name}?",
                "Who recorded the fastest lap in the {race_name}?",
                "What was the best lap time in the {race_name}?",
                "Who had the fastest lap at the {race_name}?",
                "What was the quickest lap time in the {race_name}?",
                "Who set the fastest lap of the race?",
                "What was the fastest lap in qualifying?",
                "Who got the fastest lap in practice?"
            ],
            
            "position_changes": [
                "How many positions did {driver_name} gain in the {race_name}?",
                "What was {driver_name}'s biggest position gain?",
                "How did {driver_name}'s position change during the {race_name}?",
                "What was the biggest position change in the {race_name}?",
                "How many positions did {driver_name} lose?",
                "What was {driver_name}'s position improvement?",
                "How did positions change in the {race_name}?",
                "What was the most positions gained in the {race_name}?",
                "How did {driver_name} move up the field?",
                "What was the biggest position loss in the {race_name}?",
                "How many positions did {driver_name} gain from qualifying?",
                "What was {driver_name}'s position change from start to finish?",
                "How did the field order change in the {race_name}?",
                "What was the most dramatic position change?",
                "How did {driver_name}'s position evolve during the race?"
            ],
            
            "pit_stops": [
                "How many pit stops did {driver_name} make in the {race_name}?",
                "What was {driver_name}'s fastest pit stop?",
                "How long were the pit stops in the {race_name}?",
                "What was the fastest pit stop of the {race_name}?",
                "How many pit stops did {team_name} make?",
                "What was the average pit stop time in the {race_name}?",
                "How did pit stops affect the {race_name}?",
                "What was {driver_name}'s pit stop strategy?",
                "How many pit stops were there in the {race_name}?",
                "What was the slowest pit stop in the {race_name}?",
                "How did pit stops influence the race result?",
                "What was {team_name}'s pit stop performance?",
                "How many pit stops did the winner make?",
                "What was the pit stop timing like?",
                "How did pit stops change the race order?"
            ],
            
            "tire_strategy": [
                "What tire strategy did {driver_name} use in the {race_name}?",
                "How did tire compounds affect the {race_name}?",
                "What was the best tire strategy in the {race_name}?",
                "How many tire changes did {driver_name} make?",
                "What tire compounds were used in the {race_name}?",
                "How did tire wear affect the {race_name}?",
                "What was {team_name}'s tire strategy?",
                "How did different tire compounds perform?",
                "What was the optimal tire strategy?",
                "How did tire degradation impact the race?",
                "What tire compounds lasted longest?",
                "How did tire strategy influence the result?",
                "What was the tire performance like?",
                "How did teams manage tire wear?",
                "What was the most effective tire strategy?"
            ]
        }
        
        # Positions
        self.positions = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th"]
        self.position_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        
        # Time contexts
        self.time_contexts = [
            "this season", "this year", "last race", "most recent race",
            "last weekend", "yesterday", "today", "recently", "latest race"
        ]
        
        # Schema
        self.schema = {
            "tables": {
                "positions_transformed": ["driver_number", "position", "session_key", "meeting_key", "date", "position_change", "is_leader"],
                "drivers_transformed": ["driver_number", "full_name", "team_name", "session_key", "meeting_key"],
                "sessions_transformed": ["session_key", "session_name", "date_start", "meeting_key", "session_type"],
                "meetings": ["meeting_key", "meeting_name", "country_name", "circuit_short_name", "date_start", "year"],
                "laps_transformed": ["driver_number", "lap_number", "lap_duration", "session_key", "meeting_key", "is_outlier"],
                "pit_stops_transformed": ["driver_number", "lap_number", "pit_duration", "session_key", "meeting_key"],
                "stints_transformed": ["driver_number", "compound", "lap_start", "lap_end", "session_key", "meeting_key"],
                "weather_transformed": ["air_temperature", "track_temperature", "humidity", "rainfall", "session_key", "meeting_key"],
                "intervals_transformed": ["driver_number", "gap_to_leader", "interval", "session_key", "meeting_key", "is_leader"]
            }
        }

    def get_all_drivers(self) -> List[str]:
        """Get all drivers from all teams"""
        all_drivers = []
        for drivers in self.teams_drivers.values():
            all_drivers.extend(drivers)
        return all_drivers

    def get_all_teams(self) -> List[str]:
        """Get all team names"""
        return list(self.teams_drivers.keys())

    def get_all_races(self) -> List[str]:
        """Get all race names"""
        return list(self.races.values())

    def extract_entities(self, question: str) -> Dict[str, Any]:
        """Extract entities from question"""
        entities = {
            "drivers": [],
            "teams": [],
            "races": [],
            "positions": [],
            "time_context": "recent",
            "sessions": []
        }
        
        question_lower = question.lower()
        
        # Extract drivers
        for driver in self.get_all_drivers():
            if driver.lower().replace(" ", "") in question_lower.replace(" ", ""):
                entities["drivers"].append(driver)
        
        # Extract teams
        for team in self.get_all_teams():
            if team.lower() in question_lower:
                entities["teams"].append(team)
        
        # Extract races
        for race_key, race_name in self.races.items():
            if race_key in question_lower or race_name.lower() in question_lower:
                entities["races"].append(race_name)
        
        # Extract positions
        for pos in self.positions:
            if pos.lower() in question_lower:
                entities["positions"].append(pos)
        
        # Extract time context
        for context in self.time_contexts:
            if context in question_lower:
                entities["time_context"] = context
                break
        
        # Extract sessions
        if "qualifying" in question_lower:
            entities["sessions"].append("Qualifying")
        elif "race" in question_lower and "qualifying" not in question_lower:
            entities["sessions"].append("Race")
        elif "practice" in question_lower:
            entities["sessions"].extend(["Practice 1", "Practice 2", "Practice 3"])
        
        return entities

    def generate_sql_for_intent(self, intent: str, entities: Dict[str, Any]) -> str:
        """Generate SQL based on intent and entities"""
        
        base_queries = {
            "race_results": """
                SELECT d.full_name, d.team_name, p.position, m.meeting_name, s.session_name, s.date_start
                FROM positions_transformed p
                JOIN drivers_transformed d ON p.driver_number = d.driver_number AND p.session_key = d.session_key
                JOIN sessions_transformed s ON p.session_key = s.session_key
                JOIN meetings m ON p.meeting_key = m.meeting_key
                WHERE s.session_type = 'Race'
                {filters}
                ORDER BY p.position
                LIMIT 10
            """,
            
            "qualifying_results": """
                SELECT d.full_name, d.team_name, p.position, m.meeting_name, s.date_start
                FROM positions_transformed p
                JOIN drivers_transformed d ON p.driver_number = d.driver_number AND p.session_key = d.session_key
                JOIN sessions_transformed s ON p.session_key = s.session_key
                JOIN meetings m ON p.meeting_key = m.meeting_key
                WHERE s.session_type = 'Qualifying'
                {filters}
                ORDER BY p.position
                LIMIT 10
            """,
            
            "driver_performance": """
                SELECT d.full_name, d.team_name, AVG(p.position) as avg_position, 
                       COUNT(DISTINCT s.session_key) as sessions_count, m.meeting_name
                FROM drivers_transformed d
                LEFT JOIN positions_transformed p ON d.driver_number = p.driver_number AND d.session_key = p.session_key
                LEFT JOIN sessions_transformed s ON d.session_key = s.session_key
                LEFT JOIN meetings m ON d.meeting_key = m.meeting_key
                WHERE 1=1
                {filters}
                GROUP BY d.full_name, d.team_name, m.meeting_name
                ORDER BY avg_position
                LIMIT 10
            """,
            
            "team_performance": """
                SELECT d.team_name, AVG(p.position) as avg_position, 
                       COUNT(DISTINCT s.session_key) as sessions_count, m.meeting_name
                FROM drivers_transformed d
                LEFT JOIN positions_transformed p ON d.driver_number = p.driver_number AND d.session_key = p.session_key
                LEFT JOIN sessions_transformed s ON d.session_key = s.session_key
                LEFT JOIN meetings m ON d.meeting_key = m.meeting_key
                WHERE 1=1
                {filters}
                GROUP BY d.team_name, m.meeting_name
                ORDER BY avg_position
                LIMIT 10
            """,
            
            "fastest_laps": """
                SELECT d.full_name, d.team_name, l.lap_duration, l.lap_number, m.meeting_name, s.session_name
                FROM laps_transformed l
                JOIN drivers_transformed d ON l.driver_number = d.driver_number AND l.session_key = d.session_key
                JOIN sessions_transformed s ON l.session_key = s.session_key
                JOIN meetings m ON l.meeting_key = m.meeting_key
                WHERE l.is_outlier = false
                {filters}
                ORDER BY l.lap_duration
                LIMIT 10
            """,
            
            "position_changes": """
                SELECT d.full_name, d.team_name, p.position_change, p.position, m.meeting_name, s.session_name
                FROM positions_transformed p
                JOIN drivers_transformed d ON p.driver_number = d.driver_number AND p.session_key = d.session_key
                JOIN sessions_transformed s ON p.session_key = s.session_key
                JOIN meetings m ON p.meeting_key = m.meeting_key
                WHERE p.position_change IS NOT NULL
                {filters}
                ORDER BY ABS(p.position_change) DESC
                LIMIT 10
            """,
            
            "pit_stops": """
                SELECT d.full_name, d.team_name, COUNT(ps.id) as pit_stop_count, 
                       AVG(ps.pit_duration) as avg_pit_duration, m.meeting_name, s.session_name
                FROM drivers_transformed d
                LEFT JOIN pit_stops_transformed ps ON d.driver_number = ps.driver_number AND d.session_key = ps.session_key
                LEFT JOIN sessions_transformed s ON d.session_key = s.session_key
                LEFT JOIN meetings m ON d.meeting_key = m.meeting_key
                WHERE ps.id IS NOT NULL
                {filters}
                GROUP BY d.full_name, d.team_name, m.meeting_name, s.session_name
                ORDER BY pit_stop_count DESC
                LIMIT 10
            """,
            
            "tire_strategy": """
                SELECT d.full_name, d.team_name, st.compound, COUNT(st.id) as stint_count,
                       AVG(st.lap_end - st.lap_start) as avg_stint_length, m.meeting_name
                FROM stints_transformed st
                JOIN drivers_transformed d ON st.driver_number = d.driver_number AND st.session_key = d.session_key
                JOIN sessions_transformed s ON st.session_key = s.session_key
                JOIN meetings m ON st.meeting_key = m.meeting_key
                WHERE 1=1
                {filters}
                GROUP BY d.full_name, d.team_name, st.compound, m.meeting_name
                ORDER BY d.full_name, st.compound
                LIMIT 10
            """
        }
        
        # Generate filters
        filters = []
        
        if entities.get("drivers"):
            driver_names = "', '".join(entities["drivers"])
            filters.append(f"AND d.full_name IN ('{driver_names}')")
        
        if entities.get("teams"):
            team_names = "', '".join(entities["teams"])
            filters.append(f"AND d.team_name IN ('{team_names}')")
        
        if entities.get("races"):
            race_names = "', '".join(entities["races"])
            filters.append(f"AND m.meeting_name IN ('{race_names}')")
        
        if entities.get("positions"):
            pos_numbers = [str(self.positions.index(pos) + 1) for pos in entities["positions"]]
            position_values = "', '".join(pos_numbers)
            filters.append(f"AND p.position IN ('{position_values}')")
        
        if entities.get("sessions"):
            session_names = "', '".join(entities["sessions"])
            filters.append(f"AND s.session_name IN ('{session_names}')")
        
        # Time filter
        if entities["time_context"] == "season":
            filters.append("AND EXTRACT(YEAR FROM s.date_start) = EXTRACT(YEAR FROM NOW())")
        elif entities["time_context"] == "recent":
            filters.append("AND s.date_start >= NOW() - INTERVAL '30 days'")
        
        filter_string = " ".join(filters)
        
        # Get base query for intent
        base_query = base_queries.get(intent, base_queries["race_results"])
        
        return base_query.format(filters=filter_string)

    def substitute_placeholders(self, template: str) -> List[str]:
        """Substitute placeholders in template with actual values"""
        questions = []
        
        if "{driver_name}" in template:
            for driver in self.get_all_drivers():
                questions.append(template.replace("{driver_name}", driver))
        elif "{team_name}" in template:
            for team in self.get_all_teams():
                questions.append(template.replace("{team_name}", team))
        elif "{race_name}" in template:
            for race in self.get_all_races():
                questions.append(template.replace("{race_name}", race))
        elif "{position}" in template:
            for position in self.positions:
                questions.append(template.replace("{position}", position))
        else:
            questions.append(template)
        
        return questions

    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the complete dataset"""
        dataset = []
        
        print("Generating F1 dataset...")
        
        for intent, templates in self.question_templates.items():
            print(f"Processing intent: {intent}")
            
            for template in templates:
                # Generate variations
                questions = self.substitute_placeholders(template)
                
                for question in questions:
                    # Extract entities
                    entities = self.extract_entities(question)
                    
                    # Generate SQL
                    sql = self.generate_sql_for_intent(intent, entities)
                    
                    # Create training example
                    example = {
                        "question": question,
                        "intent": intent,
                        "entities": entities,
                        "sql": sql,
                        "schema": self.schema,
                        "difficulty": self.calculate_difficulty(question, sql),
                        "metadata": {
                            "drivers_mentioned": len(entities["drivers"]),
                            "teams_mentioned": len(entities["teams"]),
                            "races_mentioned": len(entities["races"]),
                            "positions_mentioned": len(entities["positions"]),
                            "has_time_context": entities["time_context"] != "recent"
                        }
                    }
                    
                    dataset.append(example)
        
        # Add complex multi-entity examples
        complex_examples = self.generate_complex_examples()
        dataset.extend(complex_examples)
        
        # Add edge cases
        edge_cases = self.generate_edge_cases()
        dataset.extend(edge_cases)
        
        print(f"Generated {len(dataset)} examples")
        return dataset

    def generate_complex_examples(self) -> List[Dict[str, Any]]:
        """Generate complex multi-entity examples"""
        complex_examples = []
        
        # Complex templates
        complex_templates = [
            "How did {driver_name} perform compared to {driver_name2} in the {race_name}?",
            "What was the difference between {team_name} and {team_name2} in qualifying?",
            "How did {driver_name} perform in {race_name} vs {race_name2}?",
            "What was {driver_name}'s performance in {race_name} qualifying vs race?",
            "How did {team_name} drivers compare in the {race_name}?",
            "What was the battle between {driver_name} and {driver_name2} in the {race_name}?",
            "How did {driver_name} perform in different sessions of the {race_name}?",
            "What was {team_name}'s strategy in the {race_name} compared to {team_name2}?",
            "How did {driver_name} handle the {race_name} vs {race_name2}?",
            "What was the difference in performance between {driver_name} and {driver_name2}?"
        ]
        
        all_drivers = self.get_all_drivers()
        all_teams = self.get_all_teams()
        all_races = self.get_all_races()
        
        for template in complex_templates:
            if "{driver_name2}" in template:
                for driver1 in all_drivers:
                    for driver2 in all_drivers:
                        if driver1 != driver2:
                            for race in all_races:
                                question = template.replace("{driver_name}", driver1).replace("{driver_name2}", driver2).replace("{race_name}", race)
                                entities = self.extract_entities(question)
                                sql = self.generate_sql_for_intent("driver_performance", entities)
                                
                                complex_examples.append({
                                    "question": question,
                                    "intent": "driver_performance",
                                    "entities": entities,
                                    "sql": sql,
                                    "schema": self.schema,
                                    "difficulty": "complex",
                                    "metadata": {"type": "multi_driver_comparison"}
                                })
            
            elif "{team_name2}" in template:
                for team1 in all_teams:
                    for team2 in all_teams:
                        if team1 != team2:
                            for race in all_races:
                                question = template.replace("{team_name}", team1).replace("{team_name2}", team2).replace("{race_name}", race)
                                entities = self.extract_entities(question)
                                sql = self.generate_sql_for_intent("team_performance", entities)
                                
                                complex_examples.append({
                                    "question": question,
                                    "intent": "team_performance",
                                    "entities": entities,
                                    "sql": sql,
                                    "schema": self.schema,
                                    "difficulty": "complex",
                                    "metadata": {"type": "multi_team_comparison"}
                                })
        
        return complex_examples

    def generate_edge_cases(self) -> List[Dict[str, Any]]:
        """Generate edge cases and unusual questions"""
        edge_cases = []
        
        edge_templates = [
            "What happened to {driver_name} in the {race_name}?",
            "Why did {driver_name} retire from the {race_name}?",
            "What was the incident involving {driver_name}?",
            "How did the weather affect the {race_name}?",
            "What was the safety car situation in the {race_name}?",
            "How did tire degradation impact the {race_name}?",
            "What was the strategy call that won the {race_name}?",
            "How did the track conditions change during the {race_name}?",
            "What was the most dramatic moment of the {race_name}?",
            "How did the race start affect the {race_name}?",
            "What was the key overtaking move in the {race_name}?",
            "How did the pit stop timing influence the {race_name}?",
            "What was the most important decision in the {race_name}?",
            "How did the race pace compare to qualifying?",
            "What was the defining moment of the {race_name}?"
        ]
        
        all_drivers = self.get_all_drivers()
        all_races = self.get_all_races()
        
        for template in edge_templates:
            if "{driver_name}" in template:
                for driver in all_drivers:
                    for race in all_races:
                        question = template.replace("{driver_name}", driver).replace("{race_name}", race)
                        entities = self.extract_entities(question)
                        sql = self.generate_sql_for_intent("race_results", entities)
                        
                        edge_cases.append({
                            "question": question,
                            "intent": "race_results",
                            "entities": entities,
                            "sql": sql,
                            "schema": self.schema,
                            "difficulty": "edge_case",
                            "metadata": {"type": "incident_analysis"}
                        })
        
        return edge_cases

    def calculate_difficulty(self, question: str, sql: str) -> str:
        """Calculate difficulty level of the question"""
        complexity_score = 0
        
        # Count entities
        entities = self.extract_entities(question)
        complexity_score += len(entities["drivers"])
        complexity_score += len(entities["teams"])
        complexity_score += len(entities["races"])
        complexity_score += len(entities["positions"])
        
        # Check for complex patterns
        if "compare" in question.lower():
            complexity_score += 2
        if "vs" in question.lower():
            complexity_score += 2
        if "difference" in question.lower():
            complexity_score += 1
        if "strategy" in question.lower():
            complexity_score += 2
        
        # Check SQL complexity
        if "GROUP BY" in sql:
            complexity_score += 1
        if "AVG" in sql or "COUNT" in sql:
            complexity_score += 1
        if "JOIN" in sql and sql.count("JOIN") > 2:
            complexity_score += 1
        
        if complexity_score <= 2:
            return "easy"
        elif complexity_score <= 4:
            return "medium"
        else:
            return "hard"

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "f1_training_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
        
        # Print statistics
        intents = {}
        difficulties = {}
        for example in dataset:
            intent = example["intent"]
            difficulty = example["difficulty"]
            
            intents[intent] = intents.get(intent, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        print("\nDataset Statistics:")
        print(f"Total examples: {len(dataset)}")
        print("\nIntent distribution:")
        for intent, count in intents.items():
            print(f"  {intent}: {count}")
        
        print("\nDifficulty distribution:")
        for difficulty, count in difficulties.items():
            print(f"  {difficulty}: {count}")
    
    
    @staticmethod
    def sample_balanced_dataset(dataset, max_samples=2000, per_intent_limit=250, difficulty_ratio=(0.5, 0.3, 0.2)):
        """
        Returns a balanced dataset with a max of `max_samples`, limited to `per_intent_limit` per intent.
        difficulty_ratio = (easy_ratio, medium_ratio, hard_ratio)
        """
        from collections import defaultdict
        import random

        grouped = defaultdict(lambda: {'easy': [], 'medium': [], 'hard': []})

        # Group examples by intent and difficulty (normalize labels)
        for ex in dataset:
            intent = ex['intent']
            difficulty = ex['difficulty']
            if difficulty not in ['easy', 'medium', 'hard']:
                difficulty = 'hard'  # Treat 'complex' and 'edge_case' as 'hard'
            grouped[intent][difficulty].append(ex)

        final_samples = []
        easy_ratio, medium_ratio, hard_ratio = difficulty_ratio

        for intent, levels in grouped.items():
            limit = min(per_intent_limit, max_samples - len(final_samples))
            if limit <= 0:
                break

            n_easy = int(limit * easy_ratio)
            n_medium = int(limit * medium_ratio)
            n_hard = limit - n_easy - n_medium

            sampled = (
                random.sample(levels['easy'], min(n_easy, len(levels['easy']))) +
                random.sample(levels['medium'], min(n_medium, len(levels['medium']))) +
                random.sample(levels['hard'], min(n_hard, len(levels['hard'])))
            )
            final_samples.extend(sampled)

        return final_samples


def main():
    """Main function to generate the dataset"""
    # generator = F1DatasetGenerator()
    # dataset = generator.generate_dataset()
    # generator.save_dataset(dataset)
    generator = F1DatasetGenerator()
    full_dataset = generator.generate_dataset()
    sampled_dataset = F1DatasetGenerator.sample_balanced_dataset(full_dataset, max_samples=2000)
    generator.save_dataset(sampled_dataset, filename="f1_training_balanced_dataset.json")

if __name__ == "__main__":
    main()