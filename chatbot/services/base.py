from abc import abstractmethod, ABC
from typing import Sequence

class BaseModel(ABC):
    @abstractmethod
    def invoke(self, message: str, history: Sequence[dict[str, str]], context: str | None = None) -> str:
        pass
