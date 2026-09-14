@echo off
setlocal
set OUTPUT_DIR=D:\7_codespace\terminal\runs\v4_i0_structured_expert_dataset_20260910_r3
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
D:\7_codespace\terminal\venv\Scripts\python.exe -m terminal_marl.training.low_v4_imitation i0 --output "%OUTPUT_DIR%" --target-rows 1200 1>"%OUTPUT_DIR%\collector.stdout.log" 2>"%OUTPUT_DIR%\collector.stderr.log"
exit /b %ERRORLEVEL%
