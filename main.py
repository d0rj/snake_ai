from argparse import ArgumentParser
from sys import exit
from typing import Callable

from sessions.quiet_session import session
from sessions.windowed_session import session as ui_session


def import_function(module_name: str, function_name: str) -> Callable:
    try:
        module = __import__(module_name, fromlist=[function_name])
        function = getattr(module, function_name)
        return function
    except ModuleNotFoundError:
        print(f'No such module with name: \'{module_name}\'')
        exit(-1)
    except AttributeError:
        print(f'No such function with name: \'{function_name}\'')
        exit(-1)


if __name__ == '__main__':
    parser = ArgumentParser(description='Run snake session')
    parser.add_argument(
        '-n', '--no-window',
        action='store_true',
        help='(flag) Do not display game on the screen'
    )
    parser.add_argument(
        '-k', '--keyboard',
        action='store_true',
        help='(flag) Control snake by keyboard'
    )
    parser.add_argument(
        '-m', '--module',
        type=str, default='ai.snake_ai',
        help='Module name of file, where search AI function'
    )
    parser.add_argument(
        '-f', '--function',
        type=str, default='snake_ai',
        help='Name of AI function'
    )

    args = parser.parse_args()

    snake_ai = import_function(str(args.module), str(args.function))
    session_func = session if bool(args.no_window) else ui_session

    session_func(snake_ai, bool(args.keyboard))
