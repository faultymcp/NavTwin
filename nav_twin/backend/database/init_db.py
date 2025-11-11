"""
Database initialization script
Creates all tables and seeds initial data
"""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Base, SystemConfig, MLModelMetadata
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/neuronav")

def init_database():
    """Initialize database with all tables"""
    print("🔧 Initializing database...")
    
    # Create engine
    engine = create_engine(DATABASE_URL, echo=True)
    
    # Create all tables
    print("📋 Creating tables...")
    Base.metadata.create_all(bind=engine)
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Seed initial system configuration
        print("🌱 Seeding initial data...")
        seed_system_config(db)
        seed_ml_models(db)
        
        db.commit()
        print("✅ Database initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def seed_system_config(db):
    """Seed system configuration"""
    configs = [
        {
            "key": "stress_threshold_default",
            "value": {"threshold": 0.7},
            "description": "Default stress threshold for triggering interventions"
        },
        {
            "key": "crowd_prediction_enabled",
            "value": {"enabled": True},
            "description": "Enable ML-based crowd prediction"
        },
        {
            "key": "intervention_types_enabled",
            "value": {
                "breathing": True,
                "reroute": True,
                "break": True,
                "reassurance": True,
                "sensory_tip": True
            },
            "description": "Which intervention types are enabled"
        },
        {
            "key": "supported_regions",
            "value": {
                "regions": ["london", "nyc", "sf_bay"]
            },
            "description": "Regions with GTFS data support"
        },
        {
            "key": "achievement_definitions",
            "value": {
                "first_journey": {
                    "name": "First Steps",
                    "description": "Complete your first journey",
                    "icon": "🎯",
                    "target": 1
                },
                "route_master_1": {
                    "name": "Route Explorer",
                    "description": "Complete the same route 3 times",
                    "icon": "🗺️",
                    "target": 3
                },
                "route_master_2": {
                    "name": "Route Master",
                    "description": "Complete the same route 10 times",
                    "icon": "⭐",
                    "target": 10
                },
                "stress_warrior": {
                    "name": "Stress Warrior",
                    "description": "Complete 5 journeys with stress < 3",
                    "icon": "😌",
                    "target": 5
                },
                "transfer_pro": {
                    "name": "Transfer Pro",
                    "description": "Successfully navigate 3 routes with 2+ transfers",
                    "icon": "🚇",
                    "target": 3
                },
                "rush_hour_navigator": {
                    "name": "Rush Hour Navigator",
                    "description": "Complete 5 journeys during rush hour",
                    "icon": "⏰",
                    "target": 5
                },
                "week_streak": {
                    "name": "Weekly Traveler",
                    "description": "Use the app 7 days in a row",
                    "icon": "🔥",
                    "target": 7
                }
            },
            "description": "Achievement definitions for gamification"
        },
        {
            "key": "sensory_weights",
            "value": {
                "crowd": 0.3,
                "noise": 0.25,
                "visual": 0.2,
                "transfers": 0.15,
                "duration": 0.1
            },
            "description": "Default weights for sensory scoring"
        }
    ]
    
    for config_data in configs:
        config = SystemConfig(**config_data)
        db.add(config)
    
    print(f"  ✓ Added {len(configs)} system configurations")


def seed_ml_models(db):
    """Seed ML model metadata"""
    models = [
        {
            "model_name": "personal_twin_scorer",
            "version": "1.0.0",
            "file_path": "ml_models/personal_twin_v1.pkl",
            "file_size_bytes": 0,
            "training_samples": 0,
            "training_date": datetime.now(),
            "is_active": True
        },
        {
            "model_name": "crowd_predictor",
            "version": "1.0.0",
            "file_path": "ml_models/crowd_predictor_v1.pkl",
            "file_size_bytes": 0,
            "training_samples": 0,
            "training_date": datetime.now(),
            "is_active": True
        },
        {
            "model_name": "stress_detector",
            "version": "1.0.0",
            "file_path": "ml_models/stress_detector_v1.pkl",
            "file_size_bytes": 0,
            "training_samples": 0,
            "training_date": datetime.now(),
            "is_active": True
        },
        {
            "model_name": "success_predictor",
            "version": "1.0.0",
            "file_path": "ml_models/success_predictor_v1.pkl",
            "file_size_bytes": 0,
            "training_samples": 0,
            "training_date": datetime.now(),
            "is_active": True
        }
    ]
    
    for model_data in models:
        model = MLModelMetadata(**model_data)
        db.add(model)
    
    print(f"  ✓ Added {len(models)} ML model entries")


def reset_database():
    """Drop all tables and reinitialize (DANGEROUS!)"""
    print("⚠️  WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        engine = create_engine(DATABASE_URL, echo=False)
        print("🗑️  Dropping all tables...")
        Base.metadata.drop_all(bind=engine)
        print("✅ Tables dropped")
        init_database()
    else:
        print("❌ Operation cancelled")


def create_test_data():
    """Create test data for development"""
    from database.models import User, SensoryProfile, Journey, JourneyStatus
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        print("👤 Creating test user...")
        
        # Create test user
        test_user = User(
            email="test@neuronav.ai",
            username="testuser",
            hashed_password=pwd_context.hash("testpass123"),
            is_active=True
        )
        db.add(test_user)
        db.flush()
        
        # Create sensory profile
        profile = SensoryProfile(
            user_id=test_user.id,
            crowd_sensitivity=4.0,
            noise_sensitivity=3.5,
            visual_sensitivity=3.0,
            touch_sensitivity=2.5,
            transfer_anxiety=4.5,
            time_pressure_tolerance=2.0,
            preference_for_familiarity=4.0,
            preferred_times=["morning", "evening"],
            preferred_transport_modes=["train", "bus"],
            instruction_detail_level="detailed",
            prefers_visual_over_text=True,
            wants_voice_guidance=False,
            journey_count=0,
            confidence_score=0.0
        )
        db.add(profile)
        
        # Create sample journeys
        sample_journeys = [
            {
                "origin": "King's Cross Station",
                "destination": "London Bridge Station",
                "origin_lat": 51.5308,
                "origin_lng": -0.1238,
                "destination_lat": 51.5048,
                "destination_lng": -0.0863,
                "planned_start_time": datetime.now() + timedelta(hours=1),
                "planned_end_time": datetime.now() + timedelta(hours=1, minutes=25),
                "status": JourneyStatus.PLANNED,
                "transfer_count": 1,
                "total_duration_minutes": 25,
                "walking_duration_minutes": 5,
                "transport_modes": ["metro"],
                "predicted_success_probability": 0.85,
                "predicted_comfort_score": 0.75,
                "predicted_stress_level": 0.35,
                "preparation_recommendations": [
                    "Pack headphones for noise",
                    "Leave 5 minutes early",
                    "This route has one transfer"
                ]
            },
            {
                "origin": "Times Square",
                "destination": "Central Park",
                "origin_lat": 40.7580,
                "origin_lng": -73.9855,
                "destination_lat": 40.7829,
                "destination_lng": -73.9654,
                "planned_start_time": datetime.now() + timedelta(days=1),
                "planned_end_time": datetime.now() + timedelta(days=1, minutes=20),
                "status": JourneyStatus.PLANNED,
                "transfer_count": 0,
                "total_duration_minutes": 20,
                "walking_duration_minutes": 10,
                "transport_modes": ["subway"],
                "predicted_success_probability": 0.90,
                "predicted_comfort_score": 0.80,
                "predicted_stress_level": 0.25,
                "preparation_recommendations": [
                    "Good choice! Direct route with no transfers",
                    "Moderate crowd expected"
                ]
            }
        ]
        
        for journey_data in sample_journeys:
            journey = Journey(
                user_id=test_user.id,
                **journey_data,
                selected_route={},
                alternative_routes=[]
            )
            db.add(journey)
        
        db.commit()
        print("✅ Test data created successfully!")
        print("\n📧 Test user credentials:")
        print("   Email: test@neuronav.ai")
        print("   Password: testpass123")
        
    except Exception as e:
        print(f"❌ Error creating test data: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database management')
    parser.add_argument('command', choices=['init', 'reset', 'test-data'], 
                       help='Command to execute')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init_database()
    elif args.command == 'reset':
        reset_database()
    elif args.command == 'test-data':
        create_test_data()
