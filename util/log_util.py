import logging

def create_logger(logger_name, print_console = False):
    """
    Create a logger write to file logger_name.log
    :param logger_name: name of the file
    :param print_console: True = print log on console (also write to file). Default False
    :return: logger
    """
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(filename=logger_name + '.log', level=logging.DEBUG, format=FORMAT)
    logger = logging.getLogger(logger_name)
    if (print_console):
        logger.addHandler(logging.StreamHandler())
    return logger
