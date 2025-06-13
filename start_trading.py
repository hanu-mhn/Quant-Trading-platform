#!/usr/bin/env python3
"""
Quick Start Script - Get Trading in 60 Seconds!
"""

import subprocess
import time


import webbrowser
from pathlib import Path
import sys

def main():
    print("🚀 QUANTITATIVE TRADING PLATFORM - QUICK START")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("src/api/api_server.py").exists():
        print("❌ Please run this from the project root directory")
        return 1
    
    print("1️⃣ Starting API Server...")
    try:
        # Start API server in background
        api_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.api.api_server:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
        print("   ✅ API Server starting on http://localhost:8000")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        print("\n2️⃣ Starting Trading Dashboard...")
        # Start Streamlit dashboard
        dashboard_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            "src/dashboard/trading_dashboard.py",
            "--server.port", "8501",
            "--server.headless", "true"
        ])
        print("   ✅ Dashboard starting on http://localhost:8501")
        
        time.sleep(5)
        
        print("\n🎉 PLATFORM IS READY!")
        print("=" * 60)
        print("📊 Trading Dashboard: http://localhost:8501")
        print("🔧 API Documentation: http://localhost:8000/docs")
        print("❤️  Health Check: http://localhost:8000/health")
        print("=" * 60)
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8501")
            print("🌐 Opening dashboard in your browser...")
        except:
            print("💡 Manually open http://localhost:8501 in your browser")
        
        print("\n⚡ WHAT TO DO NEXT:")
        print("1. Open the dashboard to start paper trading")
        print("2. Configure your first strategy")
        print("3. Set risk parameters")
        print("4. Start virtual trading!")
        
        print("\n🛑 To stop: Press Ctrl+C")
        
        # Keep running
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            api_process.terminate()
            dashboard_process.terminate()
            print("✅ Services stopped")
            
    except Exception as e:
        print(f"❌ Error starting services: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
