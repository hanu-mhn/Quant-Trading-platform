@echo off
echo Starting Quantitative Trading Platform...
cd /d "%~dp0"
echo The app will be available at http://localhost:8501
echo.
python -m streamlit run streamlit_app.py --server.address=localhost --server.headless=false
pause
