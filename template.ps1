# PowerShell version of template.sh
New-Item -ItemType Directory -Force -Path src
New-Item -ItemType Directory -Force -Path research

New-Item -ItemType File -Force -Path src/__init__.py
New-Item -ItemType File -Force -Path src/helper.py
New-Item -ItemType File -Force -Path src/prompt.py
New-Item -ItemType File -Force -Path .env
New-Item -ItemType File -Force -Path setup.py
New-Item -ItemType File -Force -Path app.py
New-Item -ItemType File -Force -Path research/trials.ipynb
New-Item -ItemType File -Force -Path requirements.txt

Set-Content -Path README.md -Value "# Project Title"
