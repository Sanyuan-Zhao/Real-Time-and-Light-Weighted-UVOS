import logging
from pathlib import Path

import colorlog

_formatter = colorlog.ColoredFormatter('%(asctime)s %(log_color)s%(levelname)-8s%(reset)s [%(name)s] %(message)s')


def get_logger(module_name: str, log_level: int = logging.INFO):
    """
    Get a logger for this module.
    :param module_name: the path to the python file, or simply the module name
    :param log_level: messages less severe than this level get omitted
    :return: the logger
    """
    module_name = Path(module_name).stem
    logger = colorlog.getLogger(module_name)
    handler = colorlog.StreamHandler()
    handler.setFormatter(_formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    return logger


if __name__ == '__main__':
    log = get_logger(__file__, log_level=logging.DEBUG)

    log.debug('debug text')
    log.info('info text')
    log.warning('warning text')
    log.error('error text')
    log.critical('critical text')
