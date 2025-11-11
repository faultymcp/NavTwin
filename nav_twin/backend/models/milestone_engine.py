"""
MILESTONE ENGINE
Breaks down routes into gamifiable, bite-sized milestones

Turns a complex journey into:
- Walk to bus stop (5 points)
- Board bus 243 (10 points) 
- Transfer at station (15 points - harder!)
- Walk to destination (5 points)
"""

from typing import Dict, List, Any
import uuid


class MilestoneEngine:
    """
    Converts Google Routes into discrete, gamifiable milestones
    Each milestone = one completable action
    """
    
    def __init__(self):
        self.milestone_types = {
            'walk': {'base_points': 5, 'badges': ['🚶 Walker']},
            'board_transit': {'base_points': 10, 'badges': ['🚌 Transit Pro']},
            'ride_transit': {'base_points': 5, 'badges': ['🎫 Passenger']},
            'transfer': {'base_points': 20, 'badges': ['🔄 Transfer Master']},
            'exit_transit': {'base_points': 10, 'badges': ['🚪 Exit Expert']},
            'arrival': {'base_points': 25, 'badges': ['🎯 Destination Reached']},
        }
    
    def generate_milestones_from_route(self, route_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert route steps into milestones
        
        Args:
            route_data: Route from Google Maps (with legs/steps)
        
        Returns:
            List of milestone objects ready for gamification
        """
        
        milestones = []
        milestone_counter = 1
        
        # Extract legs from route
        legs = route_data.get('legs', [])
        
        if not legs:
            # No detailed route - create simple milestone
            return [{
                'id': str(uuid.uuid4()),
                'type': 'arrival',
                'title': 'Complete Journey',
                'description': 'Navigate to your destination',
                'instruction': 'Follow your route to reach the destination',
                'distance_meters': route_data.get('distance_meters', 0),
                'duration_seconds': route_data.get('duration_seconds', 0),
                'gamification': self.milestone_types['arrival'],
                'completed': False
            }]
        
        # Process each leg
        for leg_idx, leg in enumerate(legs):
            steps = leg.get('steps', [])
            
            for step_idx, step in enumerate(steps):
                travel_mode = step.get('travel_mode', 'WALK')
                
                # Determine milestone type
                if travel_mode == 'WALK':
                    milestone_type = 'walk'
                    title = f"Walk {self._format_distance(step.get('distance', {}).get('value', 0))}"
                    instruction = step.get('html_instructions', 'Walk to next location')
                
                elif travel_mode == 'TRANSIT':
                    transit_details = step.get('transit_details', {})
                    
                    # Is this boarding, riding, or exiting?
                    is_first_transit = step_idx > 0 and steps[step_idx - 1].get('travel_mode') != 'TRANSIT'
                    is_last_transit = step_idx < len(steps) - 1 and steps[step_idx + 1].get('travel_mode') != 'TRANSIT'
                    
                    if is_first_transit:
                        # Boarding transit
                        milestone_type = 'board_transit'
                        line_name = transit_details.get('line', {}).get('short_name') or transit_details.get('line', {}).get('name', 'Transit')
                        vehicle_type = transit_details.get('line', {}).get('vehicle', {}).get('type', 'bus').lower()
                        departure_stop = transit_details.get('departure_stop', {}).get('name', 'stop')
                        
                        title = f"Board {vehicle_type} {line_name}"
                        instruction = f"Board {vehicle_type} {line_name} at {departure_stop}"
                    
                    elif is_last_transit:
                        # Exiting transit
                        milestone_type = 'exit_transit'
                        arrival_stop = transit_details.get('arrival_stop', {}).get('name', 'your stop')
                        title = f"Exit at {arrival_stop}"
                        instruction = f"Exit the vehicle at {arrival_stop}"
                    
                    else:
                        # Riding transit
                        milestone_type = 'ride_transit'
                        num_stops = transit_details.get('num_stops', 0)
                        title = f"Ride for {num_stops} stops"
                        instruction = f"Stay on vehicle for {num_stops} stops"
                    
                    # Check if this is a transfer
                    if is_last_transit and step_idx < len(steps) - 2:
                        next_next_mode = steps[step_idx + 2].get('travel_mode') if step_idx + 2 < len(steps) else None
                        if next_next_mode == 'TRANSIT':
                            milestone_type = 'transfer'
                            title = "Transfer"
                            instruction = "Exit and transfer to next vehicle"
                
                else:
                    # Other modes (DRIVE, BICYCLE, etc)
                    milestone_type = 'walk'
                    title = f"{travel_mode.title()} {self._format_distance(step.get('distance', {}).get('value', 0))}"
                    instruction = step.get('html_instructions', f'Continue via {travel_mode}')
                
                # Create milestone
                milestone = {
                    'id': str(uuid.uuid4()),
                    'order': milestone_counter,
                    'type': milestone_type,
                    'title': title,
                    'description': self._clean_html(instruction),
                    'instruction': self._clean_html(instruction),
                    'distance_meters': step.get('distance', {}).get('value', 0),
                    'duration_seconds': step.get('duration', {}).get('value', 0),
                    'start_location': step.get('start_location', {}),
                    'end_location': step.get('end_location', {}),
                    'polyline': step.get('polyline', ''),
                    'transit_info': step.get('transit_details', {}),
                    'gamification': self.milestone_types.get(milestone_type, self.milestone_types['walk']),
                    'completed': False
                }
                
                milestones.append(milestone)
                milestone_counter += 1
        
        # Add final arrival milestone
        milestones.append({
            'id': str(uuid.uuid4()),
            'order': milestone_counter,
            'type': 'arrival',
            'title': '🎉 Arrive at Destination',
            'description': 'You made it!',
            'instruction': 'You have reached your destination. Great job!',
            'distance_meters': 0,
            'duration_seconds': 0,
            'gamification': self.milestone_types['arrival'],
            'completed': False
        })
        
        return milestones
    
    def _format_distance(self, meters: int) -> str:
        """Format distance for display"""
        if meters < 1000:
            return f"{meters}m"
        else:
            km = meters / 1000
            return f"{km:.1f}km"
    
    def _clean_html(self, html_text: str) -> str:
        """Remove HTML tags from instruction text"""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_text)
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_milestone_difficulty(self, milestone: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """
        Calculate how difficult this milestone is for this specific user
        
        Returns:
            Difficulty score 0-1 (0 = easy, 1 = very hard)
        """
        
        difficulty = 0.0
        
        # Base difficulty by type
        type_difficulty = {
            'walk': 0.2,
            'board_transit': 0.4,
            'ride_transit': 0.3,
            'transfer': 0.7,  # Transfers are hard!
            'exit_transit': 0.4,
            'arrival': 0.1
        }
        
        difficulty += type_difficulty.get(milestone['type'], 0.3)
        
        # Add difficulty based on user anxieties
        if milestone['type'] == 'transfer':
            transfer_anxiety = user_profile.get('transfer_anxiety', 3.0)
            difficulty += (transfer_anxiety / 5.0) * 0.3
        
        if milestone['type'] in ['board_transit', 'ride_transit']:
            crowd_sensitivity = user_profile.get('crowd_sensitivity', 3.0)
            difficulty += (crowd_sensitivity / 5.0) * 0.2
        
        # Cap at 1.0
        return min(difficulty, 1.0)