"""
NAV TWIN - FASTAPI BACKEND
Complete REST API exposing all AI functionality

This API powers the mobile app and provides personalized navigation
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import os
import json
import logging
import math
import googlemaps
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("navtwin")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:navtwin:%(message)s")

# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
from database.models import User, SensoryProfile, Journey, Route  # noqa
from database.schemas import (  # pydantic schemas for requests if you use them elsewhere
    UserCreate, UserResponse, SensoryProfileUpdate,
    JourneyCreate, JourneyResponse, RouteResponse
)
from database import models  # noqa
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# -----------------------------------------------------------------------------
# AI components
# -----------------------------------------------------------------------------
from models.ml_orchestrator_adaptive import AdaptiveMLOrchestrator
from services.maps_service import GoogleMapsService
from models.stress_detector import StressDetectionSystem
from models.milestone_engine import MilestoneEngine
from services.gamification_service import GamificationService
from services.slm_service import SLMService
from services.cv_service import CVService, MockCVService

# -----------------------------------------------------------------------------
# Initialize Google Maps client
# -----------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gmaps_client = googlemaps.Client(key="AIzaSyCcxpsmr0SaXcA0j2pDItYF8NyEwut-rm0")

# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Nav Twin API",
    description="Neuro-accessible navigation with AI personalization",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def get_db():
    """Dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Pydantic models (requests/responses)
# -----------------------------------------------------------------------------
class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int

class RouteRequest(BaseModel):
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float
    departure_time: Optional[datetime] = None
    mode: str = "TRANSIT"

class PersonalizedRoutesResponse(BaseModel):
    routes: List[Dict[str, Any]]
    personalization_applied: bool
    user_profile_summary: Dict[str, Any]

class StressReading(BaseModel):
    heart_rate: Optional[float] = None
    movement_intensity: Optional[float] = None
    dwelling_time: Optional[float] = None
    interaction_pattern: Optional[str] = None
    route_deviation: Optional[float] = None

class StressAnalysisRequest(BaseModel):
    journey_id: int
    current_data: StressReading
    current_location: Dict[str, float]

class JourneyPlanRequest(BaseModel):
    start_latitude: float
    start_longitude: float
    destination_latitude: float
    destination_longitude: float
    user_id: int
    start_address: Optional[str] = None
    destination_address: Optional[str] = None
    transport_mode: Optional[str] = "transit"  # 'walking', 'transit', 'driving'

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user (token is user_id for now)."""
    try:
        user_id = int(credentials.credentials)
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def _journey_success_ratio(journeys: List[Journey]) -> Dict[str, Any]:
    """Compute success stats from journey table using columns that actually exist."""
    total = len(journeys)
    if total == 0:
        return {"overall_success_rate": 0.85, "total_journeys": 0}
    # Our table has 'actual_success' boolean instead of 'completed_successfully'
    successes = sum(1 for j in journeys if getattr(j, "actual_success", None) is True)
    return {
        "overall_success_rate": successes / total if total else 0.0,
        "total_journeys": total,
    }

def _orchestrator_for(user: User, db: Session) -> AdaptiveMLOrchestrator:
    """Initialize ML Orchestrator with profile + history (using valid DB fields)."""

    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    profile_dict = {
        'crowd_sensitivity': profile.crowd_sensitivity,
        'noise_sensitivity': profile.noise_sensitivity,
        'visual_sensitivity': profile.visual_sensitivity,
        'touch_sensitivity': profile.touch_sensitivity,
        'transfer_anxiety': profile.transfer_anxiety,
        'time_pressure_tolerance': profile.time_pressure_tolerance,
        'preference_for_familiarity': profile.preference_for_familiarity,
        'instruction_detail_level': profile.instruction_detail_level,
        'prefers_visual_over_text': profile.prefers_visual_over_text,
        'wants_voice_guidance': profile.wants_voice_guidance
    }

    journeys = db.query(Journey).filter(Journey.user_id == user.id).all()
    hist = _journey_success_ratio(journeys)

    return AdaptiveMLOrchestrator(
        user_id=str(user.id),
        user_profile=profile_dict,
        user_history=hist,
        learned_weights=None,
        db_session=db
    )

def _ensure_gamification_fields(milestones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize milestones to always contain milestone['gamification']['points'] and ['badges'].
    """
    normalized = []
    for m in milestones or []:
        g = (m.get("gamification") or {})
        points = g.get("points", 10)  # default 10 if missing
        badges = g.get("badges", [])
        label = g.get("label", m.get("title") or m.get("name") or m.get("type") or "Milestone")

        m["gamification"] = {
            "points": points,
            "badges": badges,
            "label": label
        }
        # also ensure 'type' exists
        if not m.get("type"):
            m["type"] = "generic"
        normalized.append(m)
    return normalized

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in meters between two points using Haversine formula"""
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
                "street_name": extract_street_name(step.get('html_instructions', '')),
                "type": "navigation",
                "gamification": {
                    "points": 10,
                    "badges": [],
                    "label": f"Milestone {milestone_count}"
                }
            })
            accumulated_distance = 0
    
    return milestones

def extract_street_name(html_instruction: str) -> str:
    """Extract street name from HTML instruction"""
    # Simple extraction - can be improved with better parsing
    import re
    clean_text = re.sub('<[^<]+?>', '', html_instruction)
    words = clean_text.split()
    return words[0] if words else "ahead"

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

# -----------------------------------------------------------------------------
# Logging middleware for debugging journey start payloads
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_journey_start(request: Request, call_next):
    if request.url.path == "/api/journeys/start" and request.method == "POST":
        body_bytes = await request.body()
        try:
            body_dict = json.loads(body_bytes)
            logger.info(f"🔍 /api/journeys/start body => {json.dumps(body_dict, indent=2)}")
        except Exception as e:
            logger.error(f"Could not parse body: {e}")
        # Restore body
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive
    return await call_next(request)

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
@app.post("/api/auth/register", response_model=TokenResponse)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = User(
        email=user_data.email,
        username=user_data.username or user_data.email.split('@')[0],
        hashed_password=user_data.password  # TODO: hash in production with bcrypt
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    default_profile = SensoryProfile(user_id=new_user.id)
    db.add(default_profile)
    db.commit()

    token = str(new_user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": new_user.id}

@app.post("/api/auth/login", response_model=TokenResponse)
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == credentials.email).first()
    if not user or user.hashed_password != credentials.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = str(user.id)
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}

# -----------------------------------------------------------------------------
# User + Profile
# -----------------------------------------------------------------------------
@app.get("/api/users/me", response_model=UserResponse)
def get_current_user_info(current_user: User = Depends(_get_current_user)):
    return current_user

@app.get("/api/profile")
def get_profile(current_user: User = Depends(_get_current_user), db: Session = Depends(get_db)):
    """Get user profile with sensory data - used by Login screen"""
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    
    # Return basic user info even if no sensory profile exists
    result = {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "username": current_user.username,
        },
        "sensory_profile": {},
        "stats": {
            "total_journeys": 0,
            "total_points": 0,
            "level": 1
        }
    }
    
    # Add sensory profile if it exists
    if profile:
        result["sensory_profile"] = {
            "crowd_sensitivity": profile.crowd_sensitivity,
            "noise_sensitivity": profile.noise_sensitivity,
            "visual_sensitivity": profile.visual_sensitivity,
            "touch_sensitivity": profile.touch_sensitivity,
            "transfer_anxiety": profile.transfer_anxiety,
            "time_pressure_tolerance": profile.time_pressure_tolerance,
            "preference_for_familiarity": profile.preference_for_familiarity,
            "instruction_detail_level": profile.instruction_detail_level,
            "prefers_visual_over_text": profile.prefers_visual_over_text,
            "wants_voice_guidance": profile.wants_voice_guidance
        }
    
    return result

@app.get("/api/users/me/profile")
def get_user_profile(current_user: User = Depends(_get_current_user), db: Session = Depends(get_db)):
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "username": current_user.username,
        "sensory_profile": {
            "crowd_sensitivity": profile.crowd_sensitivity,
            "noise_sensitivity": profile.noise_sensitivity,
            "visual_sensitivity": profile.visual_sensitivity,
            "touch_sensitivity": profile.touch_sensitivity,
            "transfer_anxiety": profile.transfer_anxiety,
            "time_pressure_tolerance": profile.time_pressure_tolerance,
            "preference_for_familiarity": profile.preference_for_familiarity,
            "instruction_detail_level": profile.instruction_detail_level,
            "prefers_visual_over_text": profile.prefers_visual_over_text,
            "wants_voice_guidance": profile.wants_voice_guidance
        }
    }

@app.put("/api/users/me/profile")
def update_user_profile(
    profile_data: SensoryProfileUpdate,
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    if not profile:
        profile = SensoryProfile(user_id=current_user.id)
        db.add(profile)

    for key, value in profile_data.dict(exclude_unset=True).items():
        setattr(profile, key, value)

    db.commit()
    db.refresh(profile)
    return {"success": True, "message": "Profile updated", "profile": profile}

@app.post("/api/profile/sensory")
def create_sensory_profile(
    profile_data: dict,
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Create or update sensory profile from quiz - used by Quiz screen"""
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    if not profile:
        profile = SensoryProfile(user_id=current_user.id)
        db.add(profile)

    # Update all fields from the quiz
    for key, value in profile_data.items():
        if hasattr(profile, key):
            setattr(profile, key, value)

    db.commit()
    db.refresh(profile)
    return {"success": True, "message": "Sensory profile created", "profile": profile}

# -----------------------------------------------------------------------------
# NEW: Journey Planning Endpoint (Production Ready)
# -----------------------------------------------------------------------------
@app.post("/api/journeys/plan")
def plan_journey(request: JourneyPlanRequest, db: Session = Depends(get_db)):
    """
    Plan a neurodivergent-friendly journey with proper location search
    This replaces the old coordinate-based input with address-based planning
    """
    try:
        start_lat = request.start_latitude
        start_lng = request.start_longitude
        dest_lat = request.destination_latitude
        dest_lng = request.destination_longitude
        user_id = request.user_id
        
        logger.info(f"Planning journey for user {user_id} from ({start_lat},{start_lng}) to ({dest_lat},{dest_lng})")
        
        # Calculate total distance first
        total_distance = calculate_distance(start_lat, start_lng, dest_lat, dest_lng)
        logger.info(f"Total distance: {total_distance:.0f}m ({total_distance/1000:.2f}km)")
        
        # Log warning if journey is long, but continue planning anyway
        is_long_journey = total_distance > 3000  # 3km
        if is_long_journey:
            logger.warning(f"Long journey detected: {total_distance/1000:.1f}km - will plan anyway with appropriate milestones")
        
        # Get user's anxiety profile
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == user_id).first()
        
        # Determine anxiety level from profile
        anxiety_level = "medium"
        if profile:
            avg_sensitivity = (profile.crowd_sensitivity + profile.noise_sensitivity + profile.transfer_anxiety) / 3
            if avg_sensitivity >= 4.0:
                anxiety_level = "high"
            elif avg_sensitivity <= 2.5:
                anxiety_level = "low"
        
        logger.info(f"User anxiety level: {anxiety_level}")
        
        # Use the transport mode chosen by the user
        mode = request.transport_mode or "transit"
        logger.info(f"Using user-selected mode: {mode}")
        
        # Get route from Google Maps
        try:
            directions = gmaps_client.directions(
                origin=f"{start_lat},{start_lng}",
                destination=f"{dest_lat},{dest_lng}",
                mode=mode,
                alternatives=True,
                departure_time=datetime.now()
            )
        except Exception as e:
            logger.error(f"Google Maps API error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get route from Google Maps: {str(e)}")
        
        if not directions:
            raise HTTPException(status_code=404, detail="No route found between these locations")
        
        # Choose best route (shortest for now)
        best_route = min(directions, key=lambda r: r['legs'][0]['distance']['value'])
        
        leg = best_route['legs'][0]
        steps = leg['steps']
        
        logger.info(f"Route found: {leg['distance']['text']}, {leg['duration']['text']}, {len(steps)} steps")
        
        # Create anxiety-appropriate milestones
        milestones = create_manageable_milestones(steps, anxiety_level)
        milestones = _ensure_gamification_fields(milestones)
        
        # Prepare response
        journey_data = {
            "journey_id": f"journey_{user_id}_{int(datetime.now().timestamp())}",
            "transport_mode": mode,
            "total_distance_meters": leg['distance']['value'],
            "total_duration_seconds": leg['duration']['value'],
            "estimated_time": leg['duration']['text'],
            "start_address": request.start_address or leg['start_address'],
            "end_address": request.destination_address or leg['end_address'],
            "difficulty_rating": calculate_difficulty(leg['distance']['value'], len(steps)),
            "milestones": milestones,
            "total_milestones": len(milestones),
            "overview_polyline": best_route['overview_polyline']['points'],
            "steps": [{
                "instruction": step.get('html_instructions', ''),
                "distance": step['distance']['value'],
                "duration": step['duration']['value'],
                "start_location": {
                    "latitude": step['start_location']['lat'],
                    "longitude": step['start_location']['lng']
                },
                "end_location": {
                    "latitude": step['end_location']['lat'],
                    "longitude": step['end_location']['lng']
                },
                "maneuver": step.get('maneuver', 'straight')
            } for step in steps]
        }
        
        # Store journey in database
        try:
            journey = Journey(
                user_id=user_id,
                origin_lat=start_lat,
                origin_lng=start_lng,
                destination_lat=dest_lat,
                destination_lng=dest_lng,
                total_distance=leg['distance']['value'],
                estimated_duration=leg['duration']['value'],
                status='planned'
            )
            db.add(journey)
            db.commit()
            db.refresh(journey)
            
            # Update journey_id with the actual DB id
            journey_data["journey_id"] = journey.id
            logger.info(f"✅ Journey saved to DB with ID: {journey.id}")
            
        except Exception as e:
            logger.error(f"Failed to save journey to database: {e}")
            # Continue anyway - the journey can still be used
            logger.info(f"Journey will work without database save")
        
        return {
            "status": "success",
            "journey": journey_data,
            "anxiety_optimized": True,
            "user_anxiety_level": anxiety_level,
            "is_long_journey": is_long_journey,
            "distance_km": round(total_distance / 1000, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Journey planning error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.post("/api/routes/personalized", response_model=PersonalizedRoutesResponse)
def get_personalized_routes(
    request: RouteRequest,
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """ML-powered personalized routes for TRANSIT."""
    orchestrator = _orchestrator_for(current_user, db)

    maps_service = GoogleMapsService(api_key=os.getenv("GOOGLE_API_KEY"))
    routes = maps_service.get_routes(
        origin=(request.origin_lat, request.origin_lng),
        destination=(request.destination_lat, request.destination_lng),
        mode=request.mode,
        departure_time=request.departure_time or datetime.now(),
    )

    if not routes:
        return {"routes": [], "personalization_applied": False, "user_profile_summary": {}}

    personalized = orchestrator.rank_routes(routes)
    return {
        "routes": personalized,
        "personalization_applied": True,
        "user_profile_summary": {
            "user_id": current_user.id,
            "crowd_sensitivity": orchestrator.user_profile.get("crowd_sensitivity", 3.0),
            "transfer_anxiety": orchestrator.user_profile.get("transfer_anxiety", 3.0),
        },
    }

@app.get("/api/routes/{route_id}")
def get_route_details(
    route_id: int,
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve a single route from DB (if stored). For demonstration."""
    route = db.query(Route).filter(Route.id == route_id).first()
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route

# -----------------------------------------------------------------------------
# Journeys
# -----------------------------------------------------------------------------
@app.post("/api/journeys/start")
def start_journey(
    journey_data: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Start a new journey for the user with robust coordinate extraction."""
    logger.info(f"Starting journey for user {current_user.id}")
    logger.info(f"Journey data keys: {list(journey_data.keys())}")

    # Extract coordinates - handle both nested and flat structures
    start_lat = None
    start_lng = None
    dest_lat = None
    dest_lng = None

    # Try multiple possible structures
    if "start_location" in journey_data:
        start_loc = journey_data["start_location"]
        start_lat = start_loc.get("latitude") or start_loc.get("lat")
        start_lng = start_loc.get("longitude") or start_loc.get("lng")

    if "destination" in journey_data:
        dest_loc = journey_data["destination"]
        dest_lat = dest_loc.get("latitude") or dest_loc.get("lat")
        dest_lng = dest_loc.get("longitude") or dest_loc.get("lng")

    # Fallback to flat structure
    if start_lat is None:
        start_lat = journey_data.get("start_latitude") or journey_data.get("origin_lat")
    if start_lng is None:
        start_lng = journey_data.get("start_longitude") or journey_data.get("origin_lng")
    if dest_lat is None:
        dest_lat = journey_data.get("destination_latitude") or journey_data.get("dest_lat")
    if dest_lng is None:
        dest_lng = journey_data.get("destination_longitude") or journey_data.get("dest_lng")

    logger.info(f"Extracted coords: start=({start_lat},{start_lng}), dest=({dest_lat},{dest_lng})")

    if not all([start_lat, start_lng, dest_lat, dest_lng]):
        missing = []
        if not start_lat: missing.append("start_latitude")
        if not start_lng: missing.append("start_longitude")
        if not dest_lat: missing.append("destination_latitude")
        if not dest_lng: missing.append("destination_longitude")
        raise HTTPException(
            status_code=422,
            detail=f"Missing required coordinates: {', '.join(missing)}. Received keys: {list(journey_data.keys())}"
        )

    # Compute estimated values for now
    estimated_distance = 1000
    estimated_duration = 15 * 60

    journey = Journey(
        user_id=current_user.id,
        start_latitude=float(start_lat),
        start_longitude=float(start_lng),
        destination_latitude=float(dest_lat),
        destination_longitude=float(dest_lng),
        total_distance=estimated_distance,
        estimated_duration=estimated_duration,
        status="in_progress",
        started_at=datetime.now(),
    )
    db.add(journey)
    db.commit()
    db.refresh(journey)

    logger.info(f"✅ Journey {journey.id} created successfully")

    return {
        "journey_id": journey.id,
        "status": "in_progress",
        "message": "Journey started successfully",
        "total_distance": estimated_distance,
        "estimated_duration": estimated_duration,
    }

@app.post("/api/journeys/{journey_id}/complete")
def complete_journey(
    journey_id: int,
    completion_data: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Mark a journey as completed."""
    journey = db.query(Journey).filter(Journey.id == journey_id, Journey.user_id == current_user.id).first()
    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    journey.status = "completed"
    journey.completed_at = datetime.now()
    journey.actual_duration = completion_data.get("actual_duration", journey.estimated_duration)
    journey.actual_success = completion_data.get("success", True)

    db.commit()
    return {"success": True, "journey_id": journey_id, "message": "Journey completed"}

@app.get("/api/journeys/history")
def get_journey_history(
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Get user's journey history."""
    journeys = db.query(Journey).filter(Journey.user_id == current_user.id).order_by(Journey.created_at.desc()).all()
    return {
        "journeys": [
            {
                "id": j.id,
                "status": j.status,
                "total_distance": j.total_distance,
                "estimated_duration": j.estimated_duration,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in journeys
        ]
    }

# -----------------------------------------------------------------------------
# Stress Detection
# -----------------------------------------------------------------------------
@app.post("/api/stress/analyze")
def analyze_stress(
    request: StressAnalysisRequest,
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Real-time stress analysis during journey."""
    journey = db.query(Journey).filter(Journey.id == request.journey_id, Journey.user_id == current_user.id).first()
    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    profile_dict = {
        "crowd_sensitivity": profile.crowd_sensitivity,
        "noise_sensitivity": profile.noise_sensitivity,
        "transfer_anxiety": profile.transfer_anxiety,
    }

    detector = StressDetectionSystem(user_profile=profile_dict)

    reading = request.current_data.dict(exclude_unset=True)
    location = request.current_location

    result = detector.analyze_realtime_stress(
        current_reading=reading, journey_context={"journey_id": journey.id, "current_location": location}
    )
    return result

# -----------------------------------------------------------------------------
# Milestones + Gamification
# -----------------------------------------------------------------------------
@app.post("/api/journeys/{journey_id}/milestones")
def get_journey_milestones(
    journey_id: int,
    route_data: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Generate gamified milestones for a journey."""
    journey = db.query(Journey).filter(Journey.id == journey_id, Journey.user_id == current_user.id).first()
    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    engine = MilestoneEngine()
    raw_milestones = engine.generate_milestones_from_route(route_data)
    milestones = _ensure_gamification_fields(raw_milestones)

    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    profile_dict = {
        "crowd_sensitivity": profile.crowd_sensitivity,
        "noise_sensitivity": profile.noise_sensitivity,
        "transfer_anxiety": profile.transfer_anxiety,
        "time_pressure_tolerance": profile.time_pressure_tolerance,
    }
    user_stats = {
        "total_points": 0,
        "level": 1,
        "journeys_completed": getattr(profile, "journey_count", 0),
    }

    gamification = GamificationService(profile_dict, user_stats)
    journey_start = gamification.start_journey(milestones)

    return {"journey_id": journey_id, "milestones": milestones, "gamification": journey_start}

@app.post("/api/journeys/{journey_id}/milestones/{milestone_id}/complete")
def complete_milestone(
    journey_id: int,
    milestone_id: int,
    completion_data: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Complete a milestone and award points."""
    journey = db.query(Journey).filter(Journey.id == journey_id, Journey.user_id == current_user.id).first()
    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    milestone = completion_data.get("milestone", {})
    all_milestones = completion_data.get("all_milestones", [])
    all_milestones = _ensure_gamification_fields(all_milestones)

    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    profile_dict = {
        "crowd_sensitivity": profile.crowd_sensitivity,
        "transfer_anxiety": profile.transfer_anxiety,
        "noise_sensitivity": profile.noise_sensitivity,
        "time_pressure_tolerance": profile.time_pressure_tolerance,
    }
    user_stats = {"total_points": 0, "level": 1}

    gamification = GamificationService(profile_dict, user_stats)
    result = gamification.complete_milestone(milestone, all_milestones)
    return {"success": True, "data": result}

@app.get("/api/journeys/{journey_id}/progress")
def get_journey_progress(
    journey_id: int,
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    """Dummy progress endpoint."""
    journey = db.query(Journey).filter(Journey.id == journey_id, Journey.user_id == current_user.id).first()
    if not journey:
        raise HTTPException(status_code=404, detail="Journey not found")

    return {
        "journey_id": journey_id,
        "status": journey.status,
        "progress_percentage": 45.0,
        "completed_milestones": 3,
        "total_milestones": 7,
        "points_earned": 75,
    }

# -----------------------------------------------------------------------------
# SLM + CV
# -----------------------------------------------------------------------------
@app.post("/api/ai/instruction/generate")
def generate_ai_instruction(
    request: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    milestone = request.get("milestone")
    current_conditions = request.get("current_conditions", {})
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    profile_dict = {
        "instruction_detail_level": profile.instruction_detail_level,
        "crowd_sensitivity": profile.crowd_sensitivity,
        "noise_sensitivity": profile.noise_sensitivity,
        "transfer_anxiety": profile.transfer_anxiety,
    }
    slm = SLMService()
    instruction = slm.generate_milestone_instruction(
        milestone=milestone, user_profile=profile_dict, current_conditions=current_conditions
    )
    return {"success": True, "instruction": instruction, "personalized": True}

@app.post("/api/ai/explain-route")
def explain_route_choice(
    request: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    chosen_route = request.get("chosen_route")
    alternatives = request.get("alternative_routes", [])
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    profile_dict = {
        "crowd_sensitivity": profile.crowd_sensitivity,
        "noise_sensitivity": profile.noise_sensitivity,
        "transfer_anxiety": profile.transfer_anxiety,
    }
    slm = SLMService()
    explanation = slm.explain_route_choice(
        chosen_route=chosen_route, alternative_routes=alternatives, user_profile=profile_dict
    )
    return {"success": True, "explanation": explanation}

@app.post("/api/ai/stress-response")
def generate_stress_response(
    request: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    stress_level = request.get("stress_level", 0.3)
    stress_factors = request.get("stress_factors", [])
    current_location = request.get("current_location", {})
    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    profile_dict = {
        "crowd_sensitivity": profile.crowd_sensitivity,
        "noise_sensitivity": profile.noise_sensitivity,
        "transfer_anxiety": profile.transfer_anxiety,
    }
    slm = SLMService()
    response = slm.generate_stress_response(
        stress_level=stress_level,
        stress_factors=stress_factors,
        current_location=current_location,
        user_profile=profile_dict,
    )
    return {"success": True, "message": response, "stress_level": stress_level}

# -----------------------------------------------------------------------------
# CV
# -----------------------------------------------------------------------------
@app.post("/api/cv/analyze-scene")
def analyze_scene(
    request: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    image_data = request.get("image")
    cv = MockCVService() if not image_data else CVService()
    analysis = cv.analyze_scene(image_data)

    profile = db.query(SensoryProfile).filter(SensoryProfile.user_id == current_user.id).first()
    personalized_warnings = []
    if analysis.get("crowd_count", 0) > 8 and profile.crowd_sensitivity >= 4.0:
        personalized_warnings.append("🌟 I know crowds are hard for you. Take your time.")
    if analysis.get("brightness_level", 0) > 500 and profile.visual_sensitivity >= 4.0:
        personalized_warnings.append("😎 Very bright here. Sunglasses might help.")

    analysis["personalized_warnings"] = personalized_warnings
    return {"success": True, "analysis": analysis}

@app.post("/api/cv/validate-location")
def validate_location(
    request: Dict[str, Any],
    current_user: User = Depends(_get_current_user),
    db: Session = Depends(get_db),
):
    image_data = request.get("image")
    expected_features = request.get("expected_features", {})
    cv = CVService()
    validation = cv.validate_location(image_data, expected_features)
    return {"success": True, "validation": validation}

# -----------------------------------------------------------------------------
# Health + Root
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}

@app.get("/")
def root():
    return {
        "message": "Nav Twin API - Gamified Navigation for Neurodivergent Users",
        "version": "1.0.0",
        "docs": "/docs",
        "features": [
            "Personalized route planning",
            "Milestone-based gamification",
            "SLM-powered natural instructions",
            "Computer vision scene analysis",
            "Real-time stress detection",
        ],
    }

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)