import functools
from typing import *

_T = TypeVar('_T')
_TC = TypeVar('_TC')
_P = ParamSpec('_P')


def convert_to_collection(
    collection_builder: Callable[_P, _TC], convert_none: bool = False
):
    """A decorator to convert the result of the function.

    Args:
        collection_builder (Callable[_P, _TC]): The constructor of the collection class.
        convert_None (bool, optional): If the function returns `None`, it will return an empty collection if `convert_none` is True, otherwise return `None`. Defaults to False.
    """

    def _convert(func: Callable[_P, Optional[Iterable[_T]]]) -> Callable[_P, _TC]:

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            result: Optional[Iterable[_T]] = func(*args, **kwargs)

            if result is None:
                if convert_none:
                    return collection_builder([])
                else:
                    return None
            else:
                return collection_builder(result)

        return _wrapper

    return _convert
