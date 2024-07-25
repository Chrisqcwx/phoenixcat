import abc
import os

class WriterMixin(abc.ABC):
    def __init__(
        self,
        project: str,
        name: str,
        dir: os.PathLike
    ) -> None:
        self.project = project
        self.name = name
        self.dir = dir
    
    @abc.abstractmethod
    def log_config(self, config: dict):
        pass
    
    # @abc.abstractmethod
    # def log_data_by_step(self, name: str, value: float, step: int):
    #     pass
    
    # @abc.abstractmethod
    # def log_data_by_epoch(self, name: str, value: float, step: int):
    #     pass