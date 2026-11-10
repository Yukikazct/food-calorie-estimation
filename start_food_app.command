#!/bin/zsh
cd -- "${0:A:h}" || exit 1
exec .venv-sitting-posture/bin/python scripts/start_food_app.py --open-browser
