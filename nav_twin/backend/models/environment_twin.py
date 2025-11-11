"""
COMPONENT 2: Environment Digital Twin
Real-time transit environment modeling with crowd prediction
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone

import numpy as np


class EnvironmentDigitalTwin:
    """
    Environment Digital Twin that models real-time transit conditions
    Predicts crowd levels, noise, and sensory environment
    """

    def __init__(self):
        # Crowd patterns by hour (0-23) and day (0-6, Mon-Sun)
        self.historical_patterns: Dict[str, Any] = {}

    def predict_environment(
        self,
        route_data: Dict,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Predict environmental conditions for a route at a specific time.
        If no timestamp is provided, use "now" (UTC).
        """

        # Default timestamp
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        # If a naive datetime sneaks in, make it UTC-naive-safe
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Get time-based factors
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        is_rush_hour = self._is_rush_hour(hour, is_weekend)

        # Normalize transport modes to lowercase for downstream logic
        modes = route_data.get("transport_modes", []) or []
        route_data = dict(route_data)  # shallow copy so we don't mutate caller data
        route_data["transport_modes"] = [str(m).lower() for m in modes]

        # Predict crowd density
        crowd_density = self._predict_crowd_density(
            route_data,
            hour,
            day_of_week,
            is_rush_hour,
        )

        # Predict noise level
        noise_level = self._predict_noise_level(route_data, crowd_density)

        # Predict visual stimulus
        visual_stimulus = self._predict_visual_stimulus(route_data, crowd_density)

        # Get service alerts
        service_alerts = self._get_service_alerts(route_data)

        # Predict delays
        delay_minutes = self._predict_delays(route_data, is_rush_hour, service_alerts)

        return {
            "timestamp": timestamp.isoformat(),
            "crowd_density": float(crowd_density),
            "noise_level": float(noise_level),
            "visual_stimulus": float(visual_stimulus),
            "predicted_delay_minutes": float(delay_minutes),
            "is_rush_hour": is_rush_hour,
            "service_alerts": service_alerts,
            "confidence": self._calculate_prediction_confidence(route_data),
        }

    def _is_rush_hour(self, hour: int, is_weekend: bool) -> bool:
        """Determine if time is rush hour"""
        if is_weekend:
            return False
        # Morning rush: 7-9 AM; Evening rush: 5-7 PM
        return (7 <= hour < 9) or (17 <= hour < 19)

    def _predict_crowd_density(
        self,
        route_data: Dict,
        hour: int,
        day_of_week: int,
        is_rush_hour: bool,
    ) -> float:
        """
        Predict crowd density (0-1 scale)
        Uses time-series patterns and route characteristics
        """

        # Base crowd level by hour (typical pattern)
        hour_patterns = {
            # Night hours (0-5): very low
            0: 0.1,
            1: 0.05,
            2: 0.05,
            3: 0.05,
            4: 0.1,
            5: 0.15,
            # Morning (6-9): increasing to peak
            6: 0.3,
            7: 0.7,
            8: 0.9,
            9: 0.7,
            # Midday (10-14): moderate
            10: 0.5,
            11: 0.5,
            12: 0.6,
            13: 0.6,
            14: 0.5,
            # Afternoon (15-19): increasing to evening peak
            15: 0.6,
            16: 0.7,
            17: 0.9,
            18: 0.8,
            19: 0.6,
            # Evening (20-23): decreasing
            20: 0.4,
            21: 0.3,
            22: 0.2,
            23: 0.15,
        }

        base_crowd = hour_patterns.get(hour, 0.5)

        # Adjust for weekend (generally less crowded except leisure routes)
        is_weekend = day_of_week >= 5
        if is_weekend:
            # Weekend pattern: less in morning, more in afternoon
            if 6 <= hour < 12:
                base_crowd *= 0.6
            elif 12 <= hour < 18:
                base_crowd *= 0.8

        # Adjust for transport mode
        transport_modes = route_data.get("transport_modes", []) or []
        mode_factor = 1.0

        # Map common names to our expectations
        # 'transit' (Google) can be subway/metro/train; we bias a bit upwards
        if any(m in transport_modes for m in ["metro", "subway"]):
            mode_factor = 1.2  # Metro tends to be more crowded
        elif "train" in transport_modes:
            mode_factor = 1.1
        elif "bus" in transport_modes or "transit" in transport_modes:
            mode_factor = 1.0
        elif "walk" in transport_modes or "walking" in transport_modes:
            mode_factor = 0.6

        # Adjust for route popularity (if available)
        route_popularity = float(route_data.get("popularity_score", 0.5) or 0.5)

        # Calculate final crowd prediction
        crowd_density = base_crowd * mode_factor * (0.8 + 0.4 * route_popularity)

        # Add some randomness for realism (±10%)
        noise = float(np.random.normal(0, 0.05))
        crowd_density = float(np.clip(crowd_density + noise, 0.0, 1.0))

        return crowd_density

    def _predict_noise_level(self, route_data: Dict, crowd_density: float) -> float:
        """
        Predict noise level (0-1 scale)
        Correlated with crowd density and transport mode
        """

        transport_modes = route_data.get("transport_modes", []) or []

        # Base noise by mode
        mode_noise = {
            "metro": 0.7,
            "subway": 0.7,
            "train": 0.6,
            "bus": 0.5,
            "tram": 0.5,
            "walk": 0.2,
            "walking": 0.2,
            "transit": 0.6,  # generic transit
        }

        # Get average noise for route modes
        noise_values = [mode_noise.get(mode, 0.5) for mode in transport_modes]
        base_noise = float(np.mean(noise_values)) if noise_values else 0.5

        # Crowd contributes to noise (30% correlation)
        crowd_contribution = 0.3 * float(crowd_density)

        # Check if route goes through tunnels (increases noise)
        has_tunnels = bool(route_data.get("has_tunnels", False))
        tunnel_factor = 1.2 if has_tunnels else 1.0

        noise_level = (base_noise + crowd_contribution) * tunnel_factor
        noise_level = float(np.clip(noise_level, 0.0, 1.0))

        return noise_level

    def _predict_visual_stimulus(self, route_data: Dict, crowd_density: float) -> float:
        """
        Predict visual stimulation level (0-1 scale)
        Higher in busy stations, advertising-heavy areas
        """

        # Base visual stimulus
        base_visual = 0.4

        # Crowd increases visual stimulus
        crowd_contribution = 0.4 * float(crowd_density)

        # Urban/busy stations have more visual stimulus
        is_major_station = bool(route_data.get("is_major_station", False))
        station_factor = 1.3 if is_major_station else 1.0

        # Indoor vs outdoor
        indoor_ratio = float(route_data.get("indoor_ratio", 0.5) or 0.5)
        # Indoor tends to have more advertising, lighting
        indoor_factor = 1.0 + (0.2 * indoor_ratio)

        visual_stimulus = (base_visual + crowd_contribution) * station_factor * indoor_factor
        visual_stimulus = float(np.clip(visual_stimulus, 0.0, 1.0))

        return visual_stimulus

    def _get_service_alerts(self, route_data: Dict) -> List[Dict[str, Any]]:
        """
        Get current service alerts for the route
        In production, this would fetch from GTFS-RT service alerts
        """

        # Placeholder: would integrate with real GTFS-RT feed
        # For demo, occasionally generate random alerts
        alerts: List[Dict[str, Any]] = []

        if float(np.random.random()) < 0.1:  # 10% chance of alert
            alert_types = [
                {
                    "type": "delay",
                    "severity": "minor",
                    "message": "Minor delays due to heavy traffic",
                    "affected_stops": [],
                },
                {
                    "type": "crowding",
                    "severity": "moderate",
                    "message": "Higher than usual passenger volume",
                    "affected_stops": [],
                },
            ]
            alerts.append(dict(np.random.choice(alert_types)))
        return alerts

    def _predict_delays(
        self,
        route_data: Dict,
        is_rush_hour: bool,
        service_alerts: List[Dict],
    ) -> float:
        """
        Predict expected delays in minutes
        """

        base_delay = 0.0

        # Rush hour adds delay
        if is_rush_hour:
            base_delay += 2.0

        # Service alerts add delay
        for alert in service_alerts or []:
            if alert.get("type") == "delay":
                severity = alert.get("severity", "minor")
                if severity == "minor":
                    base_delay += 3.0
                elif severity == "moderate":
                    base_delay += 5.0
                elif severity == "major":
                    base_delay += 10.0

        # Transport mode affects delay likelihood
        transport_modes = route_data.get("transport_modes", []) or []
        if "bus" in transport_modes:
            base_delay += 1.0  # Buses more subject to traffic

        # Add small random variance
        delay = base_delay + float(np.random.normal(0, 1.0))
        return float(max(0.0, delay))

    def _calculate_prediction_confidence(self, route_data: Dict) -> float:
        """
        Calculate confidence in predictions (0-1)
        Based on data availability and route characteristics
        """

        confidence = 0.7  # Base confidence

        # Higher confidence if we have historical data
        has_historical = bool(route_data.get("has_historical_data", False))
        if has_historical:
            confidence += 0.15

        # Higher confidence for major routes
        is_major = bool(route_data.get("is_major_route", False))
        if is_major:
            confidence += 0.1

        # Lower confidence for complex routes
        transfer_count = int(route_data.get("transfer_count", 0) or 0)
        if transfer_count > 2:
            confidence -= 0.1

        return float(np.clip(confidence, 0.0, 1.0))

    def get_realtime_update(
        self,
        route_id: str,
        vehicle_positions: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Get real-time update from GTFS-RT vehicle positions
        In production, this would fetch from actual GTFS-RT feed
        """

        if vehicle_positions:
            # Calculate actual crowd from vehicle count
            vehicle_count = len(vehicle_positions)
            # Estimate crowd based on vehicles on route
            estimated_crowd = min(1.0, vehicle_count / 10)
        else:
            vehicle_count = 0
            estimated_crowd = 0.5  # Default

        return {
            "route_id": route_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vehicle_count": vehicle_count,
            "estimated_crowd_density": float(estimated_crowd),
            "data_age_seconds": 30,  # How fresh the data is
        }

    def predict_hourly_conditions(
        self,
        route_data: Dict,
        start_time: Optional[datetime],
        hours_ahead: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Predict conditions for next N hours
        Useful for planning optimal departure time
        """

        if start_time is None:
            start_time = datetime.now(timezone.utc)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        predictions: List[Dict[str, Any]] = []

        for i in range(hours_ahead):
            future_time = start_time + timedelta(hours=i)
            prediction = self.predict_environment(route_data, future_time)
            prediction["hour_offset"] = i
            predictions.append(prediction)

        return predictions

    def recommend_optimal_time(
        self,
        route_data: Dict,
        preferred_time_window: tuple,
        user_priorities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend best time to travel within a window

        Args:
            route_data: Route information
            preferred_time_window: (start_hour, end_hour) tuple
            user_priorities: Weights for different factors
                e.g., {'crowd': 0.5, 'delay': 0.3, 'noise': 0.2}
        """

        if user_priorities is None:
            user_priorities = {"crowd": 0.5, "delay": 0.3, "noise": 0.2}

        start_hour, end_hour = preferred_time_window
        now = datetime.now(timezone.utc)
        best_score = float("-inf")
        best_time: Optional[datetime] = None

        # Evaluate each hour in window
        for hour in range(start_hour, end_hour + 1):
            test_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            conditions = self.predict_environment(route_data, test_time)

            # Composite score (higher is better)
            score = (
                user_priorities.get("crowd", 0.5) * (1 - float(conditions["crowd_density"]))
                + user_priorities.get("delay", 0.3) * (1 - float(conditions["predicted_delay_minutes"]) / 15)
                + user_priorities.get("noise", 0.2) * (1 - float(conditions["noise_level"]))
            )

            if score > best_score:
                best_score = float(score)
                best_time = test_time

        return {
            "recommended_time": best_time.isoformat() if best_time else None,
            "expected_conditions": self.predict_environment(route_data, best_time) if best_time else None,
            "score": float(best_score),
        }

    def compare_routes_environmental(
        self,
        routes: List[Dict],
        timestamp: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Compare multiple routes based on environmental conditions
        Returns routes with environmental scores
        """

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        scored_routes: List[Dict] = []

        for route in routes or []:
            conditions = self.predict_environment(route, timestamp)

            # Calculate environmental comfort score
            env_score = (
                (1 - float(conditions["crowd_density"])) * 0.4
                + (1 - float(conditions["noise_level"])) * 0.3
                + (1 - float(conditions["visual_stimulus"])) * 0.2
                + (1 - float(conditions["predicted_delay_minutes"]) / 15) * 0.1
            )

            route = dict(route)
            route["environmental_conditions"] = conditions
            route["environmental_score"] = float(env_score)

            scored_routes.append(route)

        # Sort by environmental score (desc)
        scored_routes.sort(key=lambda x: x.get("environmental_score", 0.0), reverse=True)

        return scored_routes

    def get_sensory_environment_map(
        self,
        route_data: Dict,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create a detailed sensory map of the route
        Useful for visualization and user preparation
        """

        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        conditions = self.predict_environment(route_data, timestamp)

        # Break down by route segments
        segments = route_data.get("segments") or []
        segment_conditions: List[Dict[str, Any]] = []

        for segment in segments:
            segment_cond = {
                "segment_id": segment.get("id"),
                "name": segment.get("name"),
                "type": segment.get("type"),  # 'walk', 'transit', 'transfer'
                "crowd_level": float(conditions["crowd_density"]),  # Would be more granular
                "noise_level": float(conditions["noise_level"]),
                "lighting": segment.get("lighting", "moderate"),
                "indoor": bool(segment.get("indoor", False)),
                "has_seating": bool(segment.get("has_seating", False)),
                "accessibility": segment.get("accessibility", {}),
                "estimated_duration": segment.get("duration", 0),
            }

            segment_conditions.append(segment_cond)

        return {
            "overall_conditions": conditions,
            "segment_conditions": segment_conditions,
            "sensory_hotspots": self._identify_sensory_hotspots(segment_conditions),
            "recommended_coping_strategies": self._recommend_strategies(conditions),
        }

    def _identify_sensory_hotspots(self, segments: List[Dict]) -> List[Dict]:
        """Identify segments that may be challenging"""
        hotspots: List[Dict[str, Any]] = []

        for segment in segments or []:
            challenges: List[str] = []

            if float(segment.get("crowd_level", 0)) > 0.7:
                challenges.append("high_crowd")
            if float(segment.get("noise_level", 0)) > 0.7:
                challenges.append("high_noise")
            if segment.get("type") == "transfer":
                challenges.append("complex_navigation")

            if challenges:
                hotspots.append(
                    {
                        "segment": segment.get("name"),
                        "challenges": challenges,
                        "severity": "high" if len(challenges) > 1 else "moderate",
                    }
                )

        return hotspots

    def _recommend_strategies(self, conditions: Dict) -> List[str]:
        """Recommend coping strategies based on conditions"""
        strategies: List[str] = []

        if float(conditions["crowd_density"]) > 0.7:
            strategies.append("Find a corner or less crowded area to wait")
            strategies.append("Consider noise-cancelling headphones")

        if float(conditions["noise_level"]) > 0.7:
            strategies.append("Use ear protection or music")
            strategies.append("Focus on breathing exercises")

        if float(conditions["visual_stimulus"]) > 0.7:
            strategies.append("Wear sunglasses if sensitive to bright lights")
            strategies.append("Focus on a fixed point to reduce visual overload")

        if conditions.get("is_rush_hour"):
            strategies.append("Allow extra time for delays")
            strategies.append("Have a backup plan ready")

        return strategies
