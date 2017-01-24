import logging

def create_logger(logger_name, print_console = False):
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(filename=logger_name + '.log', level=logging.DEBUG, format=FORMAT)
    logger = logging.getLogger(logger_name)
    if (print_console):
        logger.addHandler(logging.StreamHandler())
    return logger
