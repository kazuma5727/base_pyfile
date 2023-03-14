import site
import time
from functools import wraps
from logging import NullHandler, getLogger
from typing import Callable

module_path = r"C:\tool\base_pyfile"
site.addsitedir(module_path)
from base_pyfile.log_setting import get_log_handler, make_logger

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def logger_timer(level: int = 10, n=1) -> Callable:
    """
    指定された logger に、関数の実行時間をログ出力するデコレーターです。

    Args:
        level: ログレベル デフォルトはdebug
        n: 実行回数 デフォルトは1

    Returns:
        デコレーター
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            関数の実行時間をログ出力するラッパー関数です。

            Args:
                *args: 位置引数
                **kwargs: キーワード引数

            Returns:
                関数の実行結果
            """
            total_time = 0
            for i in range(n):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                total_time += end_time - start_time
            avg_time = total_time / n
            if n == 1:
                logger.log(level, f"{func.__name__}: {avg_time} seconds")
            else:
                logger.log(level, f"{func.__name__} ({n}回実行の平均): {avg_time} seconds")
            return result

        return wrapper

    return decorator


def timer(func: Callable) -> Callable:
    """
    関数の実行時間をログ出力するデコレーターです。

    Args:
        func: デコレーターを適用する関数

    Returns:
        ラッパー関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        関数の実行時間をログ出力するラッパー関数です。

        Args:
            *args: 位置引数
            **kwargs: キーワード引数

        Returns:
            関数の実行結果
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


@timer
@logger_timer(n=10)
@timer
def fibonacci(n):
    def _fib(n):
        if n < 2:
            return n
        return _fib(n - 1) + _fib(n - 2)

    return _fib(n)


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    print(fibonacci(30))
