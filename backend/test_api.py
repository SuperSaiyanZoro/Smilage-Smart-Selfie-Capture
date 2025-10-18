import requests

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test all API endpoints"""
    
    print("Testing Smilage API Endpoints...\n")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\n1. Testing Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 2: Health check
    print("\n2. Testing Health Check...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 3: System info
    print("\n3. Testing System Info...")
    response = requests.get(f"{BASE_URL}/api/system-info")
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_endpoints()
