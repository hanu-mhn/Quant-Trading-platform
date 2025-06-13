"""
Integration tests for Quant Trading Platform
These tests verify the full system is working properly by making actual API calls
"""
import os
import time
import pytest
import requests


API_URL = os.environ.get("API_URL", "http://trading_app_test:8000")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://dashboard_test:8501")


def test_api_health():
    """Test that the API health endpoint returns 200 status code."""
    # Give services time to fully start up
    max_retries = 5
    retry_delay = 2
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/health")
            response.raise_for_status()  # Raises HTTPError for bad responses
            assert response.status_code == 200
            assert "status" in response.json()
            assert response.json()["status"] == "ok"
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            print(f"Connection failed (attempt {i+1}/{max_retries}), retrying in {retry_delay}s: {e}")
            if i < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def test_dashboard_accessibility():
    """Test that the Streamlit dashboard is accessible."""
    # Give services time to fully start up
    max_retries = 5
    retry_delay = 2
    
    for i in range(max_retries):
        try:
            response = requests.get(DASHBOARD_URL)
            response.raise_for_status()
            assert response.status_code == 200
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
            print(f"Connection failed (attempt {i+1}/{max_retries}), retrying in {retry_delay}s: {e}")
            if i < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def test_basic_api_version():
    """Test that the API version endpoint returns correct data."""
    response = requests.get(f"{API_URL}/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] is not None
