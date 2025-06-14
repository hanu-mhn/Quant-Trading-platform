@echo off
echo Running tests for Quantitative Trading Platform
echo.

REM Set Python path to include project root
set PYTHONPATH=%CD%

REM Install pytest if not already installed
pip install pytest > nul 2>&1

REM Run the tests with pytest, just one file at a time to avoid complex dependencies
echo Running test_utils.py...
python -m pytest tests/test_utils.py -v

echo.
echo Running test_backtesting.py...
python -m pytest tests/test_backtesting.py -v

echo.
echo Running test_live_trading.py...
python -m pytest tests/test_live_trading.py -v

echo.
if %ERRORLEVEL% EQU 0 (
    echo All tests passed!
) else (
    echo Some tests failed. See output above for details.
)

pause
