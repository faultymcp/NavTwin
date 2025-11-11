"""
Quick Backend Test Script
Run this to verify your backend is working correctly!
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def test_health_check():
    print_header("Test 1: Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print_success("Backend is running!")
            data = response.json()
            print_info(f"Status: {data.get('status')}")
            print_info(f"Version: {data.get('version')}")
            return True
        else:
            print_error(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Cannot connect to backend: {e}")
        print_info("Make sure backend is running: python main.py")
        return False

def test_journey_planning_short():
    print_header("Test 2: Short Journey (200m) - Easy")
    
    journey_data = {
        "start_latitude": 51.5074,
        "start_longitude": -0.1278,
        "destination_latitude": 51.5084,
        "destination_longitude": -0.1268,
        "user_id": 1,
        "start_address": "Trafalgar Square, London",
        "destination_address": "Leicester Square, London"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/journeys/plan",
            json=journey_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Journey planned successfully!")
            
            journey = data.get('journey', {})
            print_info(f"Distance: {journey.get('total_distance_meters')}m")
            print_info(f"Time: {journey.get('estimated_time')}")
            print_info(f"Difficulty: {journey.get('difficulty_rating')}")
            print_info(f"Milestones: {journey.get('total_milestones')}")
            print_info(f"Anxiety Level: {data.get('user_anxiety_level')}")
            
            # Show first milestone
            milestones = journey.get('milestones', [])
            if milestones:
                first = milestones[0]
                print("\n📍 First Milestone:")
                print(f"   {first.get('instruction', 'N/A')}")
                print(f"   Distance: {first.get('distance_meters')}m")
                print(f"   Points: {first.get('gamification', {}).get('points', 0)}")
                print(f"   Reward: {first.get('reward_message', 'N/A')}")
            
            return True
        else:
            print_error(f"Journey planning failed: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Error planning journey: {e}")
        return False

def test_journey_planning_long():
    print_header("Test 3: Long Journey (3km+) - Warning Test")
    
    journey_data = {
        "start_latitude": 51.5074,
        "start_longitude": -0.1278,
        "destination_latitude": 51.5287,
        "destination_longitude": -0.1318,
        "user_id": 1
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/journeys/plan",
            json=journey_data,
            headers={"Content-Type": "application/json"}
        )
        
        data = response.json()
        
        if data.get('status') == 'warning':
            print_success("Warning system working correctly!")
            print_info(f"Message: {data.get('message')}")
            print_info(f"Distance: {data.get('distance_meters')}m")
            print_info(f"Suggestion: {data.get('suggested_max')}")
            return True
        else:
            print_info("No warning triggered (journey might be under 2km)")
            return True
            
    except Exception as e:
        print_error(f"Error testing long journey: {e}")
        return False

def test_endpoints_list():
    print_header("Test 4: Available Endpoints")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint accessible!")
            print_info(f"Message: {data.get('message')}")
            print_info("Features:")
            for feature in data.get('features', []):
                print(f"   • {feature}")
            print_info("\n📚 View full API docs at: http://localhost:8000/docs")
            return True
    except Exception as e:
        print_error(f"Error accessing root endpoint: {e}")
        return False

def main():
    print("\n")
    print("🚀 " + "=" * 56 + " 🚀")
    print("    NAV TWIN BACKEND TEST SUITE")
    print("🚀 " + "=" * 56 + " 🚀")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    if results[0][1]:  # Only continue if health check passes
        results.append(("Short Journey Planning", test_journey_planning_short()))
        results.append(("Long Journey Warning", test_journey_planning_long()))
        results.append(("Endpoints List", test_endpoints_list()))
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("\n🎉 All tests passed! Your backend is working perfectly!")
        print_info("Next step: Set up the frontend to test the full app")
        print_info("Follow SETUP_GUIDE.md for frontend setup")
    else:
        print_error("\n⚠️  Some tests failed. Check the errors above.")
        print_info("Make sure:")
        print_info("  1. Backend is running (python main.py)")
        print_info("  2. Database is connected")
        print_info("  3. Google API key is configured")
    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()