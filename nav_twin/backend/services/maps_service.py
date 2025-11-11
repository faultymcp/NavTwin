"""
GOOGLE ROUTES API - REAL INTEGRATION (computeRoutes v2)
Fixed for your demo:
- Hard-coded API key (as requested)
- FieldMask includes legs/steps/polyline + instructions
- Valid TRANSIT preferences (no TRAM)
- No unsupported extraComputations like TRANSIT_FARES
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import requests
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GoogleMapsService:
    """Google Routes API v2 integration (demo-ready)"""

    def __init__(self):
        # ⚠️ Hard-coded API key (fine for a quick demo; rotate before sharing)
        self.api_key = "AIzaSyCcxpsmr0SaXcA0j2pDItYF8NyEwut-rm0"
        self.base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        print("✅ Google Routes API initialized")

    def get_routes(
        self,
        origin: Dict[str, float],
        destination: Dict[str, float],
        mode: str = "TRANSIT",
        departure_time: Optional[datetime] = None,
        alternatives: int = 3,
    ) -> List[Dict[str, Any]]:
        print("\n📍 Requesting REAL routes from Google API...")
        print(f"   Origin: {origin['lat']:.4f}, {origin['lng']:.4f}")
        print(f"   Destination: {destination['lat']:.4f}, {destination['lng']:.4f}")

        # Ask Google ONLY for the fields we need (must include nested step fields!)
        field_mask = (
            "routes.duration,"
            "routes.distanceMeters,"
            "routes.polyline.encodedPolyline,"
            "routes.routeLabels,"
            "routes.legs.startLocation,"
            "routes.legs.endLocation,"
            "routes.legs.steps.travelMode,"
            "routes.legs.steps.staticDuration,"
            "routes.legs.steps.distanceMeters,"
            "routes.legs.steps.navigationInstruction,"
            "routes.legs.steps.startLocation,"
            "routes.legs.steps.endLocation,"
            "routes.legs.steps.polyline.encodedPolyline,"
            "routes.legs.steps.transitDetails"
        )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }

        body: Dict[str, Any] = {
            "origin": {
                "location": {
                    "latLng": {"latitude": origin["lat"], "longitude": origin["lng"]}
                }
            },
            "destination": {
                "location": {
                    "latLng": {"latitude": destination["lat"], "longitude": destination["lng"]}
                }
            },
            # Valid values: DRIVE | WALK | BICYCLE | TWO_WHEELER | TRANSIT
            "travelMode": mode.upper(),
            "computeAlternativeRoutes": True,
            "routeModifiers": {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": False,
            },
            "languageCode": "en-US",
            "units": "METRIC",
        }

        # Transit preferences (valid set for Routes v2; TRAM is NOT accepted here)
        if mode.upper() == "TRANSIT":
            body["transitPreferences"] = {
                "allowedTravelModes": ["BUS", "SUBWAY", "TRAIN", "RAIL"],
                "routingPreference": "FEWER_TRANSFERS",  # or "LESS_WALKING" / "DEFAULT"
            }
            # Do NOT set extraComputations=["TRANSIT_FARES"] (not supported in computeRoutes)

        # Departure time must be present & not in the past for consistent transit results
        dt = departure_time or datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        body["departureTime"] = dt.isoformat().replace("+00:00", "Z")

        try:
            print("   🌐 Calling Google Routes API...")
            resp = requests.post(self.base_url, headers=headers, json=body, timeout=20)
            print(f"   📡 Response status: {resp.status_code}")

            if resp.status_code == 200:
                data = resp.json()
                routes = self._parse_google_response(data)
                print(f"   ✅ Received {len(routes)} real routes from Google!")
                return routes

            # Non-200: log text and fallback to mock so your demo continues
            print(f"   ❌ Google API error {resp.status_code}: {resp.text}")
            print("   ⚠️ Falling back to mock routes...")
            return self._generate_fallback_routes(origin, destination)

        except Exception as e:
            print(f"   ❌ API request failed: {e}")
            print("   ⚠️ Falling back to mock routes...")
            return self._generate_fallback_routes(origin, destination)

    def _parse_google_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize Google response to your app's expected structure,
        keeping legs/steps/polyline so the frontend can render real instructions + maps.
        """
        routes_out: List[Dict[str, Any]] = []

        for i, r in enumerate(data.get("routes", [])):
            # duration as seconds ("123s")
            duration_s = 0
            dur = r.get("duration", "0s")
            if isinstance(dur, str) and dur.endswith("s"):
                try:
                    duration_s = int(dur[:-1])
                except ValueError:
                    duration_s = 0

            legs_out: List[Dict[str, Any]] = []
            transfer_count = 0
            mode_set = set()

            for leg in r.get("legs", []):
                leg_steps: List[Dict[str, Any]] = []

                for step in leg.get("steps", []):
                    tm = step.get("travelMode", "")
                    if tm:
                        mode_set.add(tm.lower())
                    if tm == "TRANSIT":
                        transfer_count += 1

                    sd = step.get("staticDuration", "0s")
                    static_s = int(sd[:-1]) if isinstance(sd, str) and sd.endswith("s") and sd[:-1].isdigit() else 0

                    leg_steps.append({
                        "travel_mode": tm,
                        "distance": {"value": step.get("distanceMeters", 0)},
                        "duration": {"value": static_s},
                        "html_instructions": step.get("navigationInstruction", {}).get(
                            "instructions", "Continue"
                        ),
                        "start_location": step.get("startLocation", {}).get("latLng", {}),
                        "end_location": step.get("endLocation", {}).get("latLng", {}),
                        "polyline": step.get("polyline", {}).get("encodedPolyline", ""),
                        "transit_details": step.get("transitDetails"),
                    })

                legs_out.append({
                    "start_location": leg.get("startLocation", {}).get("latLng", {}),
                    "end_location": leg.get("endLocation", {}).get("latLng", {}),
                    "steps": leg_steps,
                })

            routes_out.append({
                "route_id": i + 1,
                "name": f"Route {i + 1}",
                "duration_seconds": duration_s,
                "distance_meters": r.get("distanceMeters", 0),
                # Subtract first transit leg so it doesn't count as a "transfer"
                "transfer_count": max(0, transfer_count - 1),
                "transport_modes": sorted(mode_set),
                "legs": legs_out,
                "polyline": r.get("polyline", {}).get("encodedPolyline", ""),
                "labels": r.get("routeLabels", []),
            })

        return routes_out

    def _generate_fallback_routes(
        self, origin: Dict[str, float], destination: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Simple mock routes so the demo always shows something even if Google rejects a request.
        """
        lat_diff = abs(destination["lat"] - origin["lat"])
        lng_diff = abs(destination["lng"] - origin["lng"])
        distance_km = ((lat_diff ** 2 + lng_diff ** 2) ** 0.5) * 111

        return [
            {
                "route_id": 1,
                "name": "🚌 Direct Route (Mock)",
                "duration_seconds": int(distance_km * 180),
                "distance_meters": int(distance_km * 1000),
                "transfer_count": 0,
                "transport_modes": ["bus"],
                "legs": [
                    {
                        "steps": [
                            {
                                "travel_mode": "WALK",
                                "distance": {"value": 150},
                                "duration": {"value": 120},
                                "html_instructions": "Walk to bus stop",
                                "start_location": origin,
                                "end_location": {
                                    "lat": origin["lat"] + 0.001,
                                    "lng": origin["lng"],
                                },
                            },
                            {
                                "travel_mode": "TRANSIT",
                                "distance": {"value": int(distance_km * 900)},
                                "duration": {"value": int(distance_km * 150)},
                                "html_instructions": "Take bus to destination",
                                "start_location": {
                                    "lat": origin["lat"] + 0.001,
                                    "lng": origin["lng"],
                                },
                                "end_location": destination,
                            },
                        ]
                    }
                ],
                "polyline": "",
            }
        ]
