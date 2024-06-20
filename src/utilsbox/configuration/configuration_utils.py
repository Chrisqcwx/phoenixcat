import logging

from diffusers.configuration_utils import ConfigMixin as HF_ConfigMixin

from ..decorators import Register

logger = logging.getLogger(__name__)


class ConfigMixin(HF_ConfigMixin):
    pass


def auto_cls_from_pretrained(
    register: Register, mixin_class: type, path: str, **kwargs
):

    if not hasattr(mixin_class, 'load_config'):
        raise RuntimeError(f'Class `{mixin_class}` do not has attribute `load_config`.')

    config = mixin_class.load_config(path)
    if isinstance(config, tuple):
        config = config[0]

    cls_name = config.get('_class_name', None)
    if cls_name is None:
        raise RuntimeError('`_class_name` is not contained in config.')

    try:
        cls = register[cls_name]
    except:
        raise RuntimeError(f'_class_name `{cls_name}` has not been registered.')

    logging.debug(f'Create instance for `{cls_name}`')

    return cls.from_pretrained(path, **kwargs)
