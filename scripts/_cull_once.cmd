@echo off
rem One-shot detached cull client: holds the SSE stream open for the whole
rem cull (a dead client aborts the grade at the next stream write).
cd /d "C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator"
curl.exe -s -N -X POST -H "Content-Type: application/json" -H "X-Requested-With: FirstCut" -H "Sec-Fetch-Site: same-origin" -d @body.json http://127.0.0.1:8000/api/grade/v2/stream -o cull_stream.log
