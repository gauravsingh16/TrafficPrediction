import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


class RootLoggerManager:
    MINIMAL_FORMATTER=logging.Formatter(
        "[%(levelname)s],%(message)s",
         datefmt="%Y-m-%d% %H:%M:%S",
         
         )
    
    BASIC_FORMATTER=logging.Formatter(
        "[%(asctime)s],%(levelname)s -- :%(message)s)",
         datefmt="%Y-m-%d% %H:%M:%S",
         
         )
    
    VERBOSE_FORMATTER=logging.Formatter(
        "[%(asctime)s],(module=%(module)s func=%(funcName)s) %(levelname)s -- :%(message)s)",
         datefmt="%Y-m-%d% %H:%M:%S",
         
         )
    
    def __init__(self):
        #logging.getLogger() returns the reference to a logger instance with the specified name
        #if it is provided, or root if not
        self.logger=logging.getLogger()

        self.logger.setLevel(logging.INFO)

        self.logger.handlers = []

    def configure(self,output_path:Optional[Path]=None
     ):
        """ 
            Configure the root logger's handler and verbosity
            Args:output_path(Optional,Optional): The output path used by file handler.
        """
        #setup console logging
        self.set_console_logging(self.BASIC_FORMATTER)

        #if output path is provided then initiate the file logging
        if output_path:
            self.set_file_logging(path=output_path,formatter=self.VERBOSE_FORMATTER)

    def set_console_logging(self, formatter :logging.Formatter):
        """
            Add a "StreamHandler" to the root logger (for console logging).

            formatter (logging.Formatter): the logging formatter that will be used
        """
        handler=logging.StreamHandler()

        #set logging format
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)

        #add a handler to a logger
        self.logger.addHandler(handler)

    def set_file_logging(self,path:Path , formatter: logging.Formatter):

        """
            Add a file handler to the logger(for file logging)

            args:
            path(Path): The output path.
            formatter (logging.Formatter): the logging formatter that will be used

        """
        #if parent folder do not exist create them
        filename = path / f"{datetime.now().strftime('%Y%m%d-%H%m%d')}.log"
        try:
            filename.parent.mkdir(parents=True,exist_ok=True)
            handler=logging.FileHandler(filename,'w')
        #Make sure the file exists now

            assert filename.exists()
        except Exception as e:
            logging.critical(
                f"Unable to create the file in order to store the logs"
                )
        
        #set logging format
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)

            #add a handler to a logger
        self.logger.addHandler(handler)


logger_manager=RootLoggerManager()
logger_manager.configure(Path('./logs'))
logger=logger_manager.logger

