#!/bin/bash

# Start tmux session
tmux new-session -d -s my_project

# Split the window into three panes
tmux split-window -h
tmux split-window -v

# Send commands to each pane
tmux send-keys -t 0 'poetry run uvicorn main:app --reload' C-m
tmux send-keys -t 1 'poetry run python -m src.phoenix.server.main serve' C-m
tmux send-keys -t 2 'poetry run streamlit run streamlit_app.py' C-m

# Attach to the tmux session
tmux attach-session -t my_project