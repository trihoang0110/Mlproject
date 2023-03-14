import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

logger = logging.getLogger(__name__)
handler = logging.FileHandler(logs_path)
formatter = logging.Formatter("%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.setLevel(logging.INFO)


if __name__=='__main__':
    logger.info('testing')
