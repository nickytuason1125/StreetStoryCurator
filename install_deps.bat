@echo off
REM Installer script: run from project root (double-click or cmd.exe)
python -m pip install --upgrade pip setuptools wheel
nREM Core dependencies for optimizer + LanceDB + SigLIP/OneAlign stack
python -m pip install pymoo lancedb pyarrow numpy
nREM Optional model stack (install only if you have CUDA or want CPU fallbacks)
python -m pip install open-clip-torch bitsandbytes torch torchvision
nREM Verify installsnpython -c "import pymoo, lancedb, pyarrow, numpy; print('ok')"
echo \nTo run the N=5 optimizer test:necho python street-story-curator/src/run_optimize_n5.pynecho.necho Press any key to exit...npause >nul
