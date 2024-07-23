import os
import json
import functools
import inspect
import importlib
import logging

import diffusers
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_loading_utils import (
    LOADABLE_CLASSES,
    _unwrap_model,
    ALL_IMPORTABLE_CLASSES,
)

from ..files import load_json, safe_save_as_json

logger = logging.getLogger(__name__)

_diffusers_origin_classes = set(LOADABLE_CLASSES.keys())


def update_loadable_class(
    library, class_name, save_method="save_pretrained", load_method="from_pretrained"
):

    logger.info(f"Registering {library}.{class_name} as loadable class")

    if library not in LOADABLE_CLASSES:
        LOADABLE_CLASSES[library] = {}
    LOADABLE_CLASSES[library][class_name] = [save_method, load_method]
    ALL_IMPORTABLE_CLASSES[class_name] = [save_method, load_method]


def pipeline_loadable(save_method="save_pretrained", load_method="from_pretrained"):

    def _inner_wrapper(cls):
        library, class_name = cls.__module__, cls.__name__
        update_loadable_class(library, class_name, save_method, load_method)
        return cls

    return _inner_wrapper


@pipeline_loadable()
class PipelineRecord:

    config_name = "pipeline_config.json"

    def __init__(self, **kwargs):
        self._record = kwargs

    def set(self, key, value):
        self._record[key] = value

    def get(self, key):
        return self._record.get(key, None)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    @classmethod
    def from_pretrained(cls, config_or_path: dict | str):
        if not isinstance(config_or_path, dict):
            config_or_path: str

            if not config_or_path.endswith(cls.config_name):
                config_or_path = os.path.join(config_or_path, cls.config_name)

            config_or_path = load_json(config_or_path)

        config = config_or_path
        return cls(**config)

    def save_pretrained(self, path: str):
        # print(self._record)
        path = os.path.join(path, self.config_name)
        safe_save_as_json(self._record, path)


def _fetch_class_library_tuple(module):
    # import it here to avoid circular import
    # diffusers_module = importlib.import_module(__name__.split(".")[0])
    diffusers_module = importlib.import_module("diffusers")
    pipelines = getattr(diffusers_module, "pipelines")

    # register the config from the original module, not the dynamo compiled one
    not_compiled_module = _unwrap_model(module)
    library = not_compiled_module.__module__.split(".")[0]

    # check if the module is a pipeline module
    module_path_items = not_compiled_module.__module__.split(".")
    pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

    path = not_compiled_module.__module__.split(".")
    is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

    # if library is not in LOADABLE_CLASSES, then it is a custom module.
    # Or if it's a pipeline module, then the module is inside the pipeline
    # folder so we set the library to module name.
    if is_pipeline_module:
        library = pipeline_dir
    elif library not in _diffusers_origin_classes:
        library = not_compiled_module.__module__

    # retrieve class_name
    class_name = not_compiled_module.__class__.__name__

    return (library, class_name)


def _is_loadable_module(module):
    if not hasattr(module, '__class__'):
        return False
    return module.__class__.__name__ in ALL_IMPORTABLE_CLASSES


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def register_to_pipeline_init(init):
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):

        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, PipelineMixin):
            raise RuntimeError(
                f"`@register_to_pipeline_init` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `PipelineMixin`."
            )

        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default
            for i, (name, p) in enumerate(signature.parameters.items())
            if i > 0
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in new_kwargs
            }
        )

        # Take note of the parameters that were not present in the loaded config
        # if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
        #     new_kwargs["_use_default_values"] = list(
        #         set(new_kwargs.keys()) - set(init_kwargs)
        #     )

        new_kwargs = {**config_init_kwargs, **new_kwargs}
        # getattr(self, "register_to_config")(**new_kwargs)
        # self.register_to_status(**new_kwargs)
        init(self, *args, **init_kwargs)

        for name, value in new_kwargs.items():
            if is_json_serializable(value):
                self.register_constants(**{name: value})
                # print(f'>> {name} {value.__class__.__name__}')
            else:
                # print(f'>>> {name} {value}')
                self.register_modules(**{name: value})

    return inner_init


class PipelineMixin(DiffusionPipeline):

    def __init__(self) -> None:
        super().__init__()
        self._pipeline_record = PipelineRecord()
        # self.register_modules(pipeline_record=PipelineRecord())

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        safe_serialization: bool = True,
        variant: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory, safe_serialization, variant, push_to_hub, **kwargs
        )
        self._pipeline_record.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path: dict | str):
        try:
            record = PipelineRecord.from_pretrained(pretrained_model_name_or_path)
            kwargs = record._record
        except Exception as e:
            kwargs = {}
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def register_constants(self, **kwargs):

        for name, value in kwargs.items():
            self._pipeline_record.set(name, value)
            setattr(self, name, value)

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrieve library
            if (
                module is None
                or isinstance(module, (tuple, list))
                and module[0] is None
            ):
                register_dict = {name: (None, None)}
            else:
                library, class_name = _fetch_class_library_tuple(module)
                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)
