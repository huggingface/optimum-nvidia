from typing import Protocol, TypeVar


C = TypeVar("C")


class CompressionRecipe(Protocol[C]):
    @property
    def config(self) -> C: ...
