import logging


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):
    """
    This method sets the logger.
    """

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)


    if path_log:
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

