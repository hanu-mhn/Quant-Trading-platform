"""
Streamlit App Entry Point for Deployment

This is the main entry point for the Streamlit Community Cloud deployment.

Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.
Developer: Malavath Hanmanth Nayak
Contact: hanmanthnayak.95@gmail.com
GitHub: https://github.com/hanu-mhn
Linkedin: https://www.linkedin.com/in/hanmanth-nayak-m-6bbab1263/

This software is provided under the MIT License.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Try to import and run the demo dashboard
try:
    from demo_dashboard import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    import streamlit as st
    st.error(f"Import error: {e}")
    st.info("Falling back to basic demo...")
    
    # Fallback basic demo
    st.title("🚀 Quantitative Trading Platform")
    st.success("✅ Streamlit deployment successful!")
    st.info("This is a basic fallback. The full demo should load above.")
    
    # Show basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Value", "$125,450", "+12.5%")
    with col2:
        st.metric("Total P&L", "+$8,135", "+6.9%")
    with col3:
        st.metric("Active Positions", "5", "Stocks")
    
    st.markdown("""
    ### 🎯 Platform Features
    - 📊 Real-time portfolio tracking
    - 🤖 Algorithmic trading strategies  
    - 📈 Advanced backtesting
    - 📱 Paper trading simulation
    - 🔧 REST API access
    """)
    
    # Developer information
    st.markdown("---")
    st.markdown("""
    ### 👨‍💻 Developer Information
    **Developed by**: Malavath Hanmanth Nayak  
    **Contact**: hanmanthnayak.95@gmail.com  
    **GitHub**: [@hanu-mhn](https://github.com/hanu-mhn)  
    
    **Copyright © 2025 Malavath Hanmanth Nayak. All rights reserved.**
    """)
    
    st.info("🔗 **GitHub Repository**: [View Source Code](https://github.com/hanu-mhn/quant-trading-platform)")