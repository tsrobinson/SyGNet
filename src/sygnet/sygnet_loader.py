from .sygnet_requirements import *
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def load(path_to_file):
    path_to_file = Path(path_to_file)
    if os.path.exists(path_to_file):
        normpath_to_file = os.path.normpath(path_to_file)
        return pd.read_pickle(normpath_to_file)
    else:
        logger.error("Argument `path_to_file` must contain a full directory path as an 'r' string")
        logger.error("For example: model = load(r'C:/Folder/SyGNet/SavedModel_01Jan22_1637')")
        return None
