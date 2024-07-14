import os
import hashlib
import pickle

from .constant import ASSETS_CACHE

WELCOME_ASSERT_PATH = os.path.join(ASSETS_CACHE, "welcome.pkl")


def _encode(name):
    return hashlib.md5(name.encode()).hexdigest()


def add_welcome_msg(name: str, welcome_msg: str):

    if not isinstance(welcome_msg, str):
        raise RuntimeError('The `welcome_msg` should be str.')

    if not os.path.exists(WELCOME_ASSERT_PATH):
        data = {}
    else:
        with open(WELCOME_ASSERT_PATH, 'rb') as f:
            data = pickle.load(f)

    encode_name = _encode(name)
    data[encode_name] = welcome_msg

    with open(WELCOME_ASSERT_PATH, 'wb') as f:
        pickle.dump(data, f)


def _default_welcome(name):
    print(f'Hello {name}!')


def welcome_print(name: str):

    if not os.path.exists(WELCOME_ASSERT_PATH):
        _default_welcome(name)
        return

    encode_name = _encode(name)
    with open(WELCOME_ASSERT_PATH, 'rb') as f:
        data = pickle.load(f)

    welcome_msg = data.get(encode_name, None)

    if welcome_msg is None:
        _default_welcome(name)
        return

    for msg in welcome_msg.split('\n'):
        print(msg)
