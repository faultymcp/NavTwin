from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import googlemaps
from datetime import datetime
import math

router = APIRouter()

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two points"""
    R = 6371000  # Earth's radius in meters
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    
    a = math.sin(Δφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def create_manageable_milestones(steps: list, user_anxiety_level: str = "medium") -> list:
    """Break journey into anxiety-appropriate milestones"""
    milestone_distances = {
        "low": 150,      # 150m per milestone
        "medium": 100,   # 100m per milestone
        "high": 50       # 50m per milestone
    }
    
    milestone_interval = milestone_distances.get(user_anxiety_level, 100)
    milestones = []
    accumulated_distance = 0
    milestone_count = 0
    
    for i, step in enumerate(steps):
        step_distance = step.get('distance', {}).get('value', 0)
        accumulated_distance += step_distance
        
        if accumulated_distance >= milestone_interval or i == len(steps) - 1:
            milestone_count += 1
            milestones.append({
                "milestone_number": milestone_count,
                "distance_meters": int(accumulated_distance),
                "instruction": step.get('html_instructions', ''),
                "maneuver": step.get('maneuver', 'straight'),
                "reward_message": get_reward_message(milestone_count),
                "street_name": step.get('html_instructions', '').split()[0]
            })
            accumulated_distance = 0
    
    return milestones

def get_reward_message(milestone_number: int) -> str:
    """Generate encouraging milestone messages"""
    messages = [
        "🎯 Great start! You're doing amazing!",
        "⭐ Excellent progress! Keep going!",
        "🌟 You're crushing it! Almost there!",
        "🎉 Fantastic! You're so close!",
        "🏆 Final stretch! You've got this!",
        "💪 Outstanding! Nearly at your destination!",
        "✨ Brilliant work! Just a bit more!",
        "🎊 Superb! You're almost there!"
    ]
    return messages[min(milestone_number - 1, len(messages) - 1)]

@router.post("/plan-journey")
async def plan_journey(request: Dict[str, Any], db: Session = Depends(get_db)):
    """Plan a neurodivergent-friendly journey"""
    try:
        start_lat = request.get('start_latitude')
        start_lng = request.get('start_longitude')
        dest_lat = request.get('destination_latitude')
        dest_lng = request.get('destination_longitude')
        user_id = request.get('user_id')
        
        # Calculate total distance first
        total_distance = calculate_distance(start_lat, start_lng, dest_lat, dest_lng)
        
        # Warn if journey is too long
        if total_distance > 3000:  # 3km
            return {
                "status": "warning",
                "message": f"This journey is {total_distance/1000:.1f}km. Consider shorter journeys for better experience.",
                "suggested_max": "We recommend journeys under 2km for optimal anxiety management."
            }
        
        # Get user's anxiety profile
        user = db.query(User).filter(User.id == user_id).first()
        anxiety_level = user.anxiety_level if user else "medium"
        
        # Get route from Google
        gmaps = googlemaps.Client(key='YOUR_API_KEY')
        
        # Request walking route with alternatives
        directions = gmaps.directions(
            origin=f"{start_lat},{start_lng}",
            destination=f"{dest_lat},{dest_lng}",
            mode="walking",
            alternatives=True,
            departure_time=datetime.now()
        )
        
        if not directions:
            raise HTTPException(status_code=404, detail="No route found")
        
        # Choose best route (shortest for now, but you can add anxiety scoring)
        best_route = min(directions, key=lambda r: r['legs'][0]['distance']['value'])
        
        leg = best_route['legs'][0]
        steps = leg['steps']
        
        # Create anxiety-appropriate milestones
        milestones = create_manageable_milestones(steps, anxiety_level)
        
        # Prepare response
        journey_data = {
            "journey_id": f"journey_{user_id}_{int(datetime.now().timestamp())}",
            "total_distance_meters": leg['distance']['value'],
            "total_duration_seconds": leg['duration']['value'],
            "estimated_time": f"{leg['duration']['text']}",
            "start_address": leg['start_address'],
            "end_address": leg['end_address'],
            "difficulty_rating": calculate_difficulty(leg['distance']['value'], len(steps)),
            "milestones": milestones,
            "total_milestones": len(milestones),
            "overview_polyline": best_route['overview_polyline']['points'],
            "steps": [{
                "instruction": step.get('html_instructions', ''),
                "distance": step['distance']['value'],
                "duration": step['duration']['value'],
                "start_location": step['start_location'],
                "end_location": step['end_location'],
                "maneuver": step.get('maneuver', 'straight')
            } for step in steps]
        }
        
        # Store journey in database
        journey = Journey(
            user_id=user_id,
            start_latitude=start_lat,
            start_longitude=start_lng,
            destination_latitude=dest_lat,
            destination_longitude=dest_lng,
            total_distance=leg['distance']['value'],
            estimated_duration=leg['duration']['value'],
            status='planned',
            created_at=datetime.now()
        )
        db.add(journey)
        db.commit()
        
        return {
            "status": "success",
            "journey": journey_data,
            "anxiety_optimized": True,
            "user_anxiety_level": anxiety_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_difficulty(distance_meters: int, num_turns: int) -> str:
    """Calculate journey difficulty for neurodivergent users"""
    if distance_meters < 300 and num_turns < 3:
        return "Easy 😊"
    elif distance_meters < 800 and num_turns < 5:
        return "Moderate 🙂"
    elif distance_meters < 1500 and num_turns < 8:
        return "Challenging 😐"
    else:
        return "Advanced 😓"