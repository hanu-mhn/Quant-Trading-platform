@echo off
echo Starting Streamlit Test App...
cd /d "%~dp0"
echo The app will be available at http://localhost:8501
echo.
python -m streamlit run test_streamlit.py --server.address=localhost --server.headless=false
pause
