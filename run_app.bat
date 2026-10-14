@echo off
echo ========================================================
echo   Launching emo - Editorial Multimodal Outfit Critic
echo ========================================================
echo.
CALL .\.venv\Scripts\activate.bat
streamlit run "fitcheck\Fashion AI Advisor.py"
pause

