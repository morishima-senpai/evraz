#!/bin/bash

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

install_prerequisites() {
    echo "Installing prerequisites..."
    if command_exists apt; then
        sudo apt update
        sudo apt install -y python3 python3-pip
    elif command_exists yum; then
        sudo yum install -y python3 python3-pip
    elif command_exists dnf; then
        sudo dnf install -y python3 python3-pip
    elif command_exists pacman; then
        sudo pacman -Syu --noconfirm python python-pip
    else
        echo "Unsupported package manager. Please install Python and pip manually."
        exit 1
    fi
}

install_python_tools() {
    echo "Installing Python tools: pylint, bandit, mypy..."
    pip3 install --upgrade pip
    pip3 install pylint bandit mypy
    pip3 install -r requirements.txt
}


run_python_script() {

    # celery -A analizer worker --loglevel=info -d
    SCRIPT_NAME="bot.py"
    if [[ -f "$SCRIPT_NAME" ]]; then
        echo "Running Python script: $SCRIPT_NAME"
        python3 "$SCRIPT_NAME"
    else
        echo "Python script '$SCRIPT_NAME' not found in the current directory."
        exit 1
    fi
}

main() {
    if ! command_exists python3 || ! command_exists pip3; then
        install_prerequisites
    fi
    install_python_tools
    run_python_script
    echo "Installation and script execution complete."
}


main