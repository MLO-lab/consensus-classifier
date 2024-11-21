import logging
import logging.config

from pathlib import Path

__all__ = ["ConsensusClassifier"]

init_path = Path(__file__).resolve()

# avoid circular imports
from cclf.clf import ConsensusClassifier  # noqa: E402

logging.config.fileConfig(
    init_path.parent / "log.conf",
    disable_existing_loggers=False,
)
logging.getLogger().setLevel(logging.INFO)

__version__ = "0.1.0"
