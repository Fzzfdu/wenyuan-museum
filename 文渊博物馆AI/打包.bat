@echo off
echo 正在安装依赖...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --upgrade >nul 2>&1
echo 正在打包成exe（约60秒）...
python -m PyInstaller --onefile --windowed --name="文渊博物馆AI导览" --icon=museum.ico app.py
echo 打包完成！exe在 dist 文件夹里
pause