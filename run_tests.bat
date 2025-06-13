@echo off
echo Running tests for Quantitative Trading Platform
echo.

REM Set Python path to include project root
set PYTHONPATH=%CD%

REM Run the tests with pytest
python -m pytest tests -v

echo.
if %ERRORLEVEL% EQU 0 (
    echo All tests passed!
) else (
    echo Some tests failed. See output above for details.
)

pause
