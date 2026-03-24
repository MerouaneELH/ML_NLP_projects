import logging
from functools import wraps
from datetime import datetime

# Setup logger
logging.basicConfig(
    filename="logs/api.log",  # Créer ce dossier ou ajuster le chemin
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_request(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logging.info(f"Function {func._name_} called at {datetime.now()}")
        return await func(*args, **kwargs)
    return wrapper

def log_info(message: str):
    logging.info(message)

def log_error(message: str):
    logging.error(message)