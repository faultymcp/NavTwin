"""
COMPUTER VISION SERVICE
Real-time visual analysis for navigation validation and sensory detection

Features:
- Crowd counting (how many people around)
- Scene brightness analysis (visual overload detection)
- Location validation (confirm you're at the right place)
- Obstacle detection (safety)
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from PIL import Image
import io
import base64


class CVService:
    """
    Computer Vision for NavTwin
    
    Uses lightweight models that can run on mobile devices:
    - YOLOv8-nano for person detection (crowd counting)
    - Basic CV for brightness/visual analysis
    - Future: Location matching with landmarks
    """
    
    def __init__(self):
        """Initialize CV service"""
        self.yolo_model = None
        self.use_yolo = False
        
        # Try to load YOLO
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')  # Nano model - very fast
            self.use_yolo = True
            print("✅ YOLO model loaded for crowd detection")
        except Exception as e:
            print(f"⚠️ YOLO not available: {e}")
            print("   Using basic CV only")
    
    def analyze_scene(self, image_data: str) -> Dict[str, Any]:
        """
        Analyze scene from camera image
        
        Args:
            image_data: Base64 encoded image from mobile camera
        
        Returns:
            Dictionary with:
            - crowd_count: Number of people detected
            - crowd_density: Low/Medium/High
            - brightness_level: Lux estimate
            - visual_complexity: Low/Medium/High
            - warnings: List of sensory warnings
        """
        
        # Decode image
        image = self._decode_image(image_data)
        
        # Analyze different aspects
        crowd_count = self._count_people(image)
        brightness = self._analyze_brightness(image)
        complexity = self._analyze_visual_complexity(image)
        
        # Determine crowd density
        if crowd_count < 3:
            density = "low"
        elif crowd_count < 8:
            density = "medium"
        else:
            density = "high"
        
        # Generate warnings
        warnings = []
        if crowd_count >= 10:
            warnings.append("⚠️ Very crowded - over 10 people detected")
        if brightness > 400:
            warnings.append("⚠️ Very bright - consider sunglasses")
        if complexity > 0.7:
            warnings.append("⚠️ Visually complex scene - lots of movement")
        
        return {
            'crowd_count': crowd_count,
            'crowd_density': density,
            'brightness_level': brightness,
            'visual_complexity': complexity,
            'warnings': warnings,
            'timestamp': self._get_timestamp()
        }
    
    def validate_location(
        self,
        image_data: str,
        expected_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate user is at the correct location
        
        Args:
            image_data: Current camera view
            expected_features: What we expect to see (bus number, platform, etc.)
        
        Returns:
            - confirmed: Boolean
            - confidence: 0-1
            - message: What was detected
        """
        
        # Decode image
        image = self._decode_image(image_data)
        
        # For now, basic validation
        # In production: Use OCR to detect bus numbers, platform signs, etc.
        
        # TODO: Implement text detection (OCR)
        # - Detect bus/train numbers
        # - Read platform signs
        # - Match with expected_features
        
        return {
            'confirmed': True,  # Placeholder
            'confidence': 0.7,
            'message': "Location validation not yet implemented",
            'detected_text': []
        }
    
    def detect_obstacles(self, image_data: str) -> Dict[str, Any]:
        """
        Detect obstacles in path
        Safety feature
        """
        
        image = self._decode_image(image_data)
        
        # TODO: Implement obstacle detection
        # - Construction barriers
        # - Stopped escalators
        # - Crowd blockages
        
        return {
            'obstacles_detected': False,
            'obstacles': [],
            'safe_to_proceed': True
        }
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    def _count_people(self, image: np.ndarray) -> int:
        """Count people in image using YOLO"""
        
        if not self.use_yolo:
            # Fallback: estimate from image properties
            return self._estimate_crowd_basic(image)
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image, verbose=False)
            
            # Count people (class 0 in COCO dataset)
            person_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # Person class
                        person_count += 1
            
            return person_count
        
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
            return self._estimate_crowd_basic(image)
    
    def _estimate_crowd_basic(self, image: np.ndarray) -> int:
        """Basic crowd estimation without ML"""
        
        # Simple heuristic: look at image variation
        # More variation often means more people/objects
        
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate standard deviation
        std = np.std(gray)
        
        # Higher std = more complex scene = possibly more people
        if std < 30:
            return 0
        elif std < 50:
            return 2
        elif std < 70:
            return 5
        else:
            return 8
    
    def _analyze_brightness(self, image: np.ndarray) -> float:
        """
        Analyze scene brightness
        Returns approximate lux value
        """
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate mean brightness (0-255)
        mean_brightness = np.mean(gray)
        
        # Convert to approximate lux
        # This is a rough estimate
        # 0-50 = dark (0-100 lux)
        # 50-100 = dim (100-300 lux)
        # 100-150 = normal (300-500 lux)
        # 150-200 = bright (500-1000 lux)
        # 200-255 = very bright (1000+ lux)
        
        lux_estimate = (mean_brightness / 255) * 1000
        
        return round(lux_estimate, 1)
    
    def _analyze_visual_complexity(self, image: np.ndarray) -> float:
        """
        Analyze visual complexity of scene
        High complexity = lots of patterns, movement, visual noise
        
        Returns: 0-1 score (0 = simple, 1 = complex)
        """
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate edge density using simple gradient
        grad_x = np.abs(np.diff(gray, axis=1))
        grad_y = np.abs(np.diff(gray, axis=0))
        
        # Average edge strength
        edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2
        
        # Normalize to 0-1
        complexity = min(edge_density / 50, 1.0)
        
        return round(complexity, 2)
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
        
        except Exception as e:
            print(f"❌ Image decode error: {e}")
            # Return blank image
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# ============================================================================
# MOCK CV SERVICE (For testing without images)
# ============================================================================

class MockCVService:
    """Mock CV service for testing"""
    
    def analyze_scene(self, image_data: str = None) -> Dict[str, Any]:
        """Return mock analysis"""
        import random
        
        crowd_count = random.randint(0, 12)
        
        return {
            'crowd_count': crowd_count,
            'crowd_density': 'low' if crowd_count < 3 else 'medium' if crowd_count < 8 else 'high',
            'brightness_level': random.randint(200, 600),
            'visual_complexity': round(random.uniform(0.2, 0.8), 2),
            'warnings': ['⚠️ Crowded area'] if crowd_count > 8 else [],
            'timestamp': self._get_timestamp()
        }
    
    def validate_location(self, image_data: str, expected_features: Dict) -> Dict:
        return {
            'confirmed': True,
            'confidence': 0.85,
            'message': 'Mock validation - location confirmed'
        }
    
    def detect_obstacles(self, image_data: str) -> Dict:
        return {
            'obstacles_detected': False,
            'obstacles': [],
            'safe_to_proceed': True
        }
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()