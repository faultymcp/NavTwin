from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, User, SensoryProfile, Journey
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Simple hash function (for testing only!)
def simple_hash(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

try:
    # Check if user exists
    existing = db.query(User).filter(User.email == "test@navtwin.ai").first()
    if existing:
        print("✅ Test user already exists!")
    else:
        # Create test user
        user = User(
            email="test@navtwin.ai",
            username="testuser",
            hashed_password=simple_hash("testpass123"),
            is_active=True
        )
        db.add(user)
        db.flush()
        
        # Create sensory profile
        profile = SensoryProfile(
            user_id=user.id,
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
        db.commit()
        
        print("✅ Test user created successfully!")
        print(f"   Email: test@navtwin.ai")
        print(f"   Password: testpass123")
        
except Exception as e:
    print(f"❌ Error: {e}")
    db.rollback()
finally:
    db.close()