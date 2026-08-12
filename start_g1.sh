#!/bin/bash

source /opt/ros/foxy/setup.bash

source ~/Projects/voice_of_unitree/.venv/bin/activate

export PYTHONPATH="$HOME/Projects/unitree_sdk2_python:/home/unitree/.local/lib/python3.8/site-packages:$PYTHONPATH"

cd ~/Projects/voice_of_unitree/g1_voice_assistant

python3 main.py
