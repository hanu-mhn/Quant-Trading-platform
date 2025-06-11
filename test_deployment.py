"""
Test script to verify deployment readiness
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports for deployment"""
    print("🧪 Testing Deployment Dependencies...")
    
    try:
        import streamlit as st
        print("✅ Streamlit - OK")
    except ImportError:
        print("❌ Streamlit - MISSING")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas - OK")
    except ImportError:
        print("❌ Pandas - MISSING")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy - OK")
    except ImportError:
        print("❌ NumPy - MISSING")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly - OK")
    except ImportError:
        print("❌ Plotly - MISSING")
        return False
    
    return True

def test_files():
    """Test required files exist"""
    print("\n📁 Testing Required Files...")
    
    required_files = [
        "streamlit_app.py",
        "demo_dashboard.py", 
        "requirements_streamlit.txt",
        ".streamlit/config.toml",
        "README.md",
        "my_first_strategy.py"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} - EXISTS")
        else:
            print(f"❌ {file} - MISSING")
            all_exist = False
    
    return all_exist

def test_demo_dashboard():
    """Test demo dashboard can be imported"""
    print("\n🎯 Testing Demo Dashboard...")
    
    try:
        from demo_dashboard import main, generate_sample_data, get_sample_positions
        print("✅ Demo dashboard imports - OK")
        
        # Test data generation
        data = generate_sample_data()
        positions = get_sample_positions()
        
        print(f"✅ Sample data generated - {len(data)} records")
        print(f"✅ Sample positions - {len(positions)} positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo dashboard - ERROR: {e}")
        return False

def test_strategy():
    """Test strategy file"""
    print("\n🤖 Testing Strategy...")
    
    try:
        from my_first_strategy import SimpleMAStrategy
        
        strategy = SimpleMAStrategy()
        info = strategy.get_strategy_info()
        
        print(f"✅ Strategy loaded - {info['name']}")
        print(f"✅ Strategy type - {info['type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy - ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 DEPLOYMENT READINESS TEST")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_files, 
        test_demo_dashboard,
        test_strategy
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for deployment!")
        print("\n🚀 Next Steps:")
        print("1. Install Git: https://git-scm.com/download/windows")
        print("2. Create GitHub repository (PUBLIC)")
        print("3. Push code to GitHub")
        print("4. Deploy on Streamlit Cloud: https://share.streamlit.io")
        return True
    else:
        print(f"⚠️  {passed}/{total} tests passed. Fix issues before deployment.")
        return False

if __name__ == "__main__":
    main()
