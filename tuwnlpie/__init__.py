__version__ = "0.0.1"

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s"
)
