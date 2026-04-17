@echo off
echo ============================================
echo   CrackSense SHM Dashboard
echo ============================================
call venv\Scripts\activate
streamlit run app.py --server.port 8501 --server.headless false
pause
