import sys
import logging
logger = logging.getLogger(__name__)

def initialize_logger(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter("%(asctime)s %(module)s %(levelname)-7s %(message)s", "%Y/%b/%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))