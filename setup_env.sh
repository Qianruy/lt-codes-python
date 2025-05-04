#!/bin/bash

# Usage: bash setup_env.sh <env_name> <requirements_file>

ENV_NAME=$1
REQ_FILE=$2

if [ -z "$ENV_NAME" ] || [ -z "$REQ_FILE" ]; then
  echo "❗ Usage: bash setup_env.sh <env_name> <requirements.txt>"
  exit 1
fi


ENV_PATH="$HOME/$ENV_NAME"
/opt/homebrew/bin/python3 -m venv "$ENV_PATH" --copies
source "$ENV_PATH/bin/activate"

pip install --upgrade pip
pip install -r $REQ_FILE

pip install ipykernel
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"

echo "Complete: Python ($ENV_NAME) kernel available!"
