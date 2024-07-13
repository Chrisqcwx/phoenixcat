import os
import hashlib
import pickle

from .constant import ASSETS_CACHE

WELCOME_ASSERT_PATH = os.path.join(ASSETS_CACHE, "welcome.pkl")


def _encode(name):
    return hashlib.md5(name.encode()).hexdigest()


def add_welcome_msgs(name, welcome_msgs):
    if not os.path.exists(WELCOME_ASSERT_PATH):
        data = {}
    else:
        with open(WELCOME_ASSERT_PATH, 'rb') as f:
            data = pickle.load(f)

    encode_name = _encode(name)
    data[encode_name] = welcome_msgs

    with open(WELCOME_ASSERT_PATH, 'wb') as f:
        pickle.dump(data)


def _default_welcome(name):
    print(f'Hello {name} !')


def welcome(name: str):

    if not os.path.exists(WELCOME_ASSERT_PATH):
        _default_welcome(name)
        return

    encode_name = _encode(name)
    with open(WELCOME_ASSERT_PATH, 'rb') as f:
        data = pickle.load(f)

    welcome_msgs = data.get(encode_name, None)

    if welcome_msgs is None:
        _default_welcome(name)
        return

    for msg in welcome_msgs:
        print(msg)
