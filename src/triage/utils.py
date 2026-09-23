"""Small shared helpers. retry and chunks copied verbatim from exercises/day2.py."""

import asyncio
import functools
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Any


def retry(
    times: int, exceptions: tuple[type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator factory. Retry up to `times` attempts on listed exceptions only.

    Re-raise the last exception after the final attempt. Preserve the wrapped
    function's name. No sleeping.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times - 1:
                        raise
            return None

        return wrapper

    return decorator


def chunks[T](items: Sequence[T], size: int) -> Iterator[list[T]]:
    """Yield consecutive lists of up to `size` items. Must be a generator.

    size < 1 raises ValueError.
    """
    if size < 1:
        raise ValueError("size must be >= 1")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


async def classify_all[T, R](
    items: Sequence[T],
    func: Callable[[T], Awaitable[R]],
    max_concurrency: int,
) -> list[R]:
    """Run func over every item, at most max_concurrency at once.

    Results are returned in input order.
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(item: T) -> R:
        async with semaphore:
            return await func(item)

    return await asyncio.gather(*(run(item) for item in items))
