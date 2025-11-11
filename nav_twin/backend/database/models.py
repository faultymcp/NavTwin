"""
Database models for NeuroNav AI system
SQLAlchemy ORM models for PostgreSQL
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, JSON, Text, Enum, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class SensitivityLevel(enum.Enum):
    """Enum for sensitivity ratings"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


class JourneyStatus(enum.Enum):
    """Journey lifecycle status"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class InterventionType(enum.Enum):
    """Types of stress interventions"""
    BREATHING_EXERCISE = "breathing"
    ROUTE_CHANGE = "reroute"
    BREAK_SUGGESTION = "break"
    REASSURANCE = "reassurance"
    SENSORY_TIP = "sensory_tip"


# ============================================================================
# COMPONENT 1: PERSONAL DIGITAL TWIN
# ============================================================================

class User(Base):
    """User account and basic info"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    sensory_profile = relationship("SensoryProfile", back_populates="user", uselist=False)
    journeys = relationship("Journey", back_populates="user")
    stress_logs = relationship("StressLog", back_populates="user")
    achievements = relationship("Achievement", back_populates="user")


class SensoryProfile(Base):
    """Personal Digital Twin - User's sensory profile (Component 1)"""
    __tablename__ = "sensory_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Core sensory sensitivities (1-5 scale)
    crowd_sensitivity = Column(Float, default=3.0)
    noise_sensitivity = Column(Float, default=3.0)
    visual_sensitivity = Column(Float, default=3.0)
    touch_sensitivity = Column(Float, default=3.0)
    
    # Travel-specific preferences
    transfer_anxiety = Column(Float, default=3.0)
    time_pressure_tolerance = Column(Float, default=3.0)
    preference_for_familiarity = Column(Float, default=3.0)
    
    # Learned preferences (JSON)
    preferred_times = Column(JSON, default=list)  # ["morning", "evening"]
    avoided_routes = Column(JSON, default=list)   # [route_ids]
    preferred_transport_modes = Column(JSON, default=list)  # ["bus", "train"]
    
    # Thresholds
    stress_threshold = Column(Float, default=0.7)  # When to trigger interventions
    max_transfer_count = Column(Integer, default=2)
    max_journey_duration = Column(Integer, default=60)  # minutes
    
    # Communication preferences
    instruction_detail_level = Column(String(20), default="moderate")  # minimal, moderate, detailed
    prefers_visual_over_text = Column(Boolean, default=True)
    wants_voice_guidance = Column(Boolean, default=False)
    
    # Profile metadata
    confidence_score = Column(Float, default=0.0)  # How well-trained the twin is (0-1)
    journey_count = Column(Integer, default=0)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sensory_profile")


class LearnedPreference(Base):
    """Tracks specific learned preferences over time"""
    __tablename__ = "learned_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    preference_type = Column(String(50))  # "route", "time", "transport_mode"
    preference_value = Column(String(255))
    confidence = Column(Float)  # How sure we are (0-1)
    learned_from_journeys = Column(Integer)  # Number of journeys that contributed
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# ============================================================================
# COMPONENT 2: ENVIRONMENT DIGITAL TWIN
# ============================================================================

class Route(Base):
    """Transit route information"""
    __tablename__ = "routes"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(100), unique=True, index=True)  # From GTFS
    
    name = Column(String(255))
    transport_mode = Column(String(50))  # bus, train, metro, tram
    region = Column(String(100))  # "london", "nyc", "sf_bay"
    
    # Static characteristics
    typical_duration = Column(Integer)  # minutes
    stop_count = Column(Integer)
    
    # Environmental characteristics (JSON)
    characteristics = Column(JSON)  # {indoor_ratio, has_escalators, etc.}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EnvironmentSnapshot(Base):
    """Real-time environment state (Component 2)"""
    __tablename__ = "environment_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False)
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Real-time metrics
    crowd_density = Column(Float)  # 0-1 scale
    predicted_crowd_density = Column(Float)  # Our prediction
    actual_crowd_density = Column(Float, nullable=True)  # Actual if reported
    
    # Environmental factors
    vehicle_count = Column(Integer)
    delay_minutes = Column(Float, default=0.0)
    service_alerts = Column(JSON, default=list)
    
    # External data
    weather_condition = Column(String(50), nullable=True)
    temperature = Column(Float, nullable=True)
    air_quality_index = Column(Integer, nullable=True)
    
    # Sensory predictions
    predicted_noise_level = Column(Float)  # 0-1 scale
    predicted_visual_stimulus = Column(Float)  # 0-1 scale
    
    # Data source
    data_source = Column(String(50))  # "gtfs_rt", "tfl_api", "mta_api"


class TransitStop(Base):
    """Transit stops/stations"""
    __tablename__ = "transit_stops"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(100), unique=True, index=True)
    
    name = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    region = Column(String(100))
    
    # Accessibility features
    has_elevator = Column(Boolean, default=False)
    has_escalator = Column(Boolean, default=False)
    is_wheelchair_accessible = Column(Boolean, default=False)
    has_shelter = Column(Boolean, default=False)
    
    # Sensory characteristics
    typical_noise_level = Column(Float)  # 0-1 scale
    is_indoor = Column(Boolean, default=False)
    has_seating = Column(Boolean, default=False)
    
    characteristics = Column(JSON)  # Additional features


# ============================================================================
# COMPONENT 3: AI-POWERED WAYFINDING
# ============================================================================

class NavigationInstruction(Base):
    """Turn-by-turn navigation instructions"""
    __tablename__ = "navigation_instructions"
    
    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=False)
    
    step_number = Column(Integer)
    instruction_type = Column(String(50))  # "walk", "board", "transfer", "exit"
    
    # Different detail levels
    instruction_simple = Column(Text)  # "Get on Bus 42"
    instruction_detailed = Column(Text)  # "Walk to the bus stop. Look for..."
    
    # Visual aids
    landmark_photo_url = Column(String(500), nullable=True)
    ar_overlay_data = Column(JSON, nullable=True)  # AR markers
    
    # Location
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Estimated timing
    estimated_duration_seconds = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SignageDetection(Base):
    """Computer vision detections from signs"""
    __tablename__ = "signage_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Image data
    image_url = Column(String(500))
    detected_text = Column(Text)  # OCR output
    simplified_text = Column(Text)  # Simplified by NLP
    
    # Detection metadata
    confidence = Column(Float)
    detected_objects = Column(JSON)  # ["sign", "arrow", "platform_number"]
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


class ConversationLog(Base):
    """Conversational assistant interactions (Component 3)"""
    __tablename__ = "conversation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=True)
    
    user_message = Column(Text)
    assistant_response = Column(Text)
    context = Column(JSON)  # Journey state when asked
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================================
# COMPONENT 4: STRESS DETECTION & INTERVENTION
# ============================================================================

class StressLog(Base):
    """Stress detection data (Component 4)"""
    __tablename__ = "stress_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=True)
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Stress measurement
    self_reported_stress = Column(Float, nullable=True)  # 1-5 scale
    predicted_stress = Column(Float, nullable=True)  # ML prediction 0-1
    
    # Behavioral signals
    app_check_frequency = Column(Float, nullable=True)  # Checks per minute
    navigation_hesitation_count = Column(Integer, default=0)
    back_button_presses = Column(Integer, default=0)
    
    # Contextual factors
    journey_progress = Column(Float)  # 0-1
    current_crowd_level = Column(Float, nullable=True)
    minutes_until_arrival = Column(Integer, nullable=True)
    
    # Wearable data (if available)
    heart_rate = Column(Integer, nullable=True)
    heart_rate_variability = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="stress_logs")
    intervention = relationship("Intervention", back_populates="stress_log", uselist=False)


class Intervention(Base):
    """Stress interventions triggered (Component 4)"""
    __tablename__ = "interventions"
    
    id = Column(Integer, primary_key=True, index=True)
    stress_log_id = Column(Integer, ForeignKey("stress_logs.id"), nullable=False)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=False)
    
    intervention_type = Column(Enum(InterventionType), nullable=False)
    
    # Intervention content
    title = Column(String(255))
    description = Column(Text)
    action_data = Column(JSON)  # Specific action parameters
    
    # User response
    was_accepted = Column(Boolean, nullable=True)
    was_helpful = Column(Boolean, nullable=True)
    user_feedback = Column(Text, nullable=True)
    
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    stress_log = relationship("StressLog", back_populates="intervention")


# ============================================================================
# COMPONENT 5: PREDICTIVE JOURNEY SUCCESS MODEL
# ============================================================================

class Journey(Base):
    """Complete journey record (integrates all components)"""
    __tablename__ = "journeys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Journey details
    origin = Column(String(255))
    destination = Column(String(255))
    origin_lat = Column(Float)
    origin_lng = Column(Float)
    destination_lat = Column(Float)
    destination_lng = Column(Float)
    
    # Timing
    planned_start_time = Column(DateTime(timezone=True))
    actual_start_time = Column(DateTime(timezone=True), nullable=True)
    planned_end_time = Column(DateTime(timezone=True))
    actual_end_time = Column(DateTime(timezone=True), nullable=True)
    
    # Status
    status = Column(Enum(JourneyStatus), default=JourneyStatus.PLANNED)
    
    # Route information (JSON)
    selected_route = Column(JSON)  # Full route details from Google Maps
    alternative_routes = Column(JSON, default=list)
    
    # Predictions (Component 5)
    predicted_success_probability = Column(Float)  # 0-1
    predicted_comfort_score = Column(Float)  # 0-1
    predicted_stress_level = Column(Float)  # 0-1
    
    # Actual outcomes
    actual_success = Column(Boolean, nullable=True)  # Did they complete it?
    actual_comfort_rating = Column(Float, nullable=True)  # User reported 1-5
    actual_stress_rating = Column(Float, nullable=True)  # User reported 1-5
    
    # Route characteristics
    transfer_count = Column(Integer)
    total_duration_minutes = Column(Integer)
    walking_duration_minutes = Column(Integer)
    transport_modes = Column(JSON, default=list)  # ["bus", "train"]
    
    # Environmental context
    weather_at_start = Column(String(50), nullable=True)
    crowd_level_prediction = Column(Float, nullable=True)
    
    # Preparation
    preparation_recommendations = Column(JSON, default=list)
    user_followed_recommendations = Column(Boolean, nullable=True)
    
    # Feedback
    user_notes = Column(Text, nullable=True)
    what_went_well = Column(JSON, default=list)
    what_was_difficult = Column(JSON, default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="journeys")
    instructions = relationship("NavigationInstruction", back_populates="journey")
    interventions = relationship("Intervention", back_populates="journey")


class JourneyPredictionLog(Base):
    """Log of prediction accuracy for model improvement"""
    __tablename__ = "journey_prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    journey_id = Column(Integer, ForeignKey("journeys.id"), nullable=False)
    
    # Input features used
    features = Column(JSON)
    
    # Predictions made
    predicted_success = Column(Float)
    predicted_comfort = Column(Float)
    predicted_stress = Column(Float)
    
    # Actual outcomes
    actual_success = Column(Boolean, nullable=True)
    actual_comfort = Column(Float, nullable=True)
    actual_stress = Column(Float, nullable=True)
    
    # Model metadata
    model_version = Column(String(50))
    prediction_confidence = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Achievement(Base):
    """Gamification achievements (Component 5)"""
    __tablename__ = "achievements"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    achievement_type = Column(String(100))  # "first_journey", "route_master"
    achievement_name = Column(String(255))
    description = Column(Text)
    
    # Progress
    current_progress = Column(Integer, default=0)
    target_progress = Column(Integer)
    is_completed = Column(Boolean, default=False)
    
    # Metadata
    icon_name = Column(String(100))
    earned_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="achievements")


class RouteMastery(Base):
    """Track user's mastery of specific routes"""
    __tablename__ = "route_mastery"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Route definition
    origin = Column(String(255))
    destination = Column(String(255))
    route_hash = Column(String(100), index=True)  # Hash of route for matching
    
    # Mastery metrics
    completion_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)  # 0-1
    average_stress = Column(Float, default=0.0)
    average_comfort = Column(Float, default=0.0)
    
    # Learning progression
    mastery_level = Column(Integer, default=0)  # 0-5 stars
    first_completed = Column(DateTime(timezone=True), nullable=True)
    last_completed = Column(DateTime(timezone=True), nullable=True)
    
    # Next challenge
    suggested_next_route = Column(JSON, nullable=True)


# ============================================================================
# SUPPORTING TABLES
# ============================================================================

class SystemConfig(Base):
    """System configuration and feature flags"""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True)
    value = Column(JSON)
    description = Column(Text)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class MLModelMetadata(Base):
    """Track ML model versions and performance"""
    __tablename__ = "ml_model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100))  # "stress_detector", "crowd_predictor"
    version = Column(String(50))
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # Model file info
    file_path = Column(String(500))
    file_size_bytes = Column(Integer)
    
    # Training info
    training_samples = Column(Integer)
    training_date = Column(DateTime(timezone=True))
    
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Add relationship back-references
Journey.instructions = relationship("NavigationInstruction", back_populates="journey")
Journey.interventions = relationship("Intervention", back_populates="journey")

NavigationInstruction.journey = relationship("Journey", back_populates="instructions")
Intervention.journey = relationship("Journey", back_populates="interventions")
