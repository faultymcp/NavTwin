"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: Optional[str] = None


class UserCreate(UserBase):
    """User registration schema"""
    password: str


class UserResponse(UserBase):
    """User response schema"""
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# SENSORY PROFILE SCHEMAS
# ============================================================================

class SensoryProfileUpdate(BaseModel):
    """Update sensory profile"""
    crowd_sensitivity: Optional[float] = None
    noise_sensitivity: Optional[float] = None
    visual_sensitivity: Optional[float] = None
    touch_sensitivity: Optional[float] = None
    transfer_anxiety: Optional[float] = None
    time_pressure_tolerance: Optional[float] = None
    preference_for_familiarity: Optional[float] = None
    instruction_detail_level: Optional[str] = None
    prefers_visual_over_text: Optional[bool] = None
    wants_voice_guidance: Optional[bool] = None


# ============================================================================
# JOURNEY SCHEMAS
# ============================================================================

class JourneyCreate(BaseModel):
    """Create new journey"""
    origin_lat: float
    origin_lng: float
    origin_address: Optional[str] = None
    destination_lat: float
    destination_lng: float
    destination_address: Optional[str] = None
    departure_time: Optional[datetime] = None


class JourneyResponse(BaseModel):
    """Journey response"""
    id: int
    user_id: int
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float
    planned_departure: Optional[datetime]
    actual_departure: Optional[datetime]
    actual_arrival: Optional[datetime]
    status: str
    completed_successfully: Optional[bool]
    user_rating: Optional[int]
    
    class Config:
        from_attributes = True


# ============================================================================
# ROUTE SCHEMAS
# ============================================================================

class RouteResponse(BaseModel):
    """Route response schema"""
    route_id: str
    duration_seconds: int
    distance_meters: int
    transfer_count: int
    transport_modes: List[str]
    final_score: Optional[float] = None
    rank: Optional[int] = None
    ranking_explanation: Optional[Dict[str, Any]] = None
    scores: Optional[Dict[str, float]] = None
    personalization_applied: Optional[bool] = None
    
    class Config:
        from_attributes = True