import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append('services')

from maps_service import GoogleMapsService
from datetime import datetime

print("="*50)
print("Testing Google Routes API")
print("="*50)

try:
    service = GoogleMapsService()
    print(f"✅ Service initialized")
    print(f"   Using Routes API: {service.use_routes_api}")
    
    # King's Cross to London Bridge
    origin = {"lat": 51.5308, "lng": -0.1238}
    destination = {"lat": 51.5048, "lng": -0.0863}
    
    print(f"\n📍 Testing route: King's Cross → London Bridge")
    
    routes = service.get_routes(
        origin=origin,
        destination=destination,
        mode="TRANSIT",
        departure_time=datetime.now(),
        alternatives=3
    )
    
    if routes:
        print(f"\n✅ SUCCESS! Found {len(routes)} routes\n")
        for i, route in enumerate(routes, 1):
            print(f"Route {i}:")
            print(f"  Duration: {route['duration_seconds']/60:.1f} minutes")
            print(f"  Distance: {route['distance_meters']/1000:.1f} km")
            print(f"  Modes: {', '.join(route['transport_modes'])}")
            print(f"  Transfers: {route['transfer_count']}")
            print(f"  Source: {route['source']}")
            print()
        
        print("="*50)
        print("🎉 Routes API is working!")
        print("="*50)
    else:
        print("\n❌ No routes found")
        print("Check:")
        print("  1. Billing enabled in Google Cloud?")
        print("  2. Routes API enabled?")
        print("  3. API key correct in .env?")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()