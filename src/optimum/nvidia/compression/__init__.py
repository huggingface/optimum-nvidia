from typing import TypeVar, Protocol


C = TypeVar("C")
class CompressionRecipe(Protocol[C]):
    @property
    def config(self) -> C:
        ...



