@echo off
cd /d "C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\frontend\src-tauri\resources"
"C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\venv\Scripts\python.exe" -m http.server 8900 --bind 127.0.0.1 > "..\..\..\http_server.log" 2>&1