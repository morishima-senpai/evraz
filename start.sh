source ./venv/bin/activate
celery -A analizer worker --loglevel=info -d
python3 bot.py