from ..decorators import Register
from diffusers.configuration_utils import ConfigMixin as HF_ConfigMixin


class ConfigMixin(HF_ConfigMixin):
    pass


def auto_cls_from_pretrained(register: Register, path: str, **kwargs):

    config = ConfigMixin.load_config(path)
    if isinstance(config, tuple):
        config = config[0]

    cls_name = config.get('_class_name', None)
    if cls_name is None:
        raise RuntimeError('`_class_name` is not contained in config.')

    try:
        cls = register[cls_name]
    except:
        raise RuntimeError(f'_class_name `{cls_name}` has not been registered.')

    return cls.from_pretrained(path, **kwargs)
