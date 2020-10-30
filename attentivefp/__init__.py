# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler
logging.getLogger(__name__).addHandler(NullHandler())
# ... Clean up.
del NullHandler

# disable rdkit logging
import rdkit
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
