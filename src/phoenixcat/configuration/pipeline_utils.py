import os
import json
import copy
import functools
import inspect
import importlib
import logging

import torch
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


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


@pipeline_loadable()
class PipelineRecord:

    config_name = "pipeline_config.json"

    def __init__(self, **kwargs):
        self._constant = {}
        self._module = {}
        for name, value in kwargs.items():
            self.set(name, value)

    def set(self, key, value):
        # self._record[key] = value
        if is_json_serializable(value):
            self._constant[key] = value
        else:
            self._module[key] = value

    def get(self, key):
        # return self._record.get(key, None)
        if key in self._constant:
            return self._constant[key]
        else:
            return self._module.get(key, None)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        init_kwargs = cls.load(pretrained_model_name_or_path)
        return cls(**init_kwargs)

    @staticmethod
    def load(pretrained_model_name_or_path: str):
        config_path = os.path.join(
            pretrained_model_name_or_path, PipelineRecord.config_name
        )

        config = load_json(config_path)

        _module = {
            key: torch.load(os.path.join(pretrained_model_name_or_path, f'{key}.pt'))
            for key in config.pop('_module')
        }

        init_kwargs = {**config, **_module}

        return init_kwargs

    def save_pretrained(self, path: str):
        # print(self._record)
        config_path = os.path.join(path, self.config_name)

        save_constant = copy.deepcopy(self._constant)
        save_constant['_module'] = list(self._module.keys())
        safe_save_as_json(save_constant, config_path)
        for name, value in self._module.items():
            torch.save(value, os.path.join(path, f'{name}.pt'))


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
            # if is_json_serializable(value):
            #     self.register_constants(**{name: value})
            #     # print(f'>> {name} {value.__class__.__name__}')
            # else:
            #     # print(f'>>> {name} {value}')
            #     self.register_modules(**{name: value})
            if hasattr(value, 'from_pretrained') and hasattr(value, 'save_pretrained'):
                self.register_modules(**{name: value})
            else:
                self.register_custom_values(**{name: value})

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
        record_path = os.path.join(save_directory, 'record')
        self._pipeline_record.save_pretrained(record_path)

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path: str):
        record_path = os.path.join(pretrained_model_name_or_path, 'record')
        try:
            kwargs = PipelineRecord.load(record_path)
        except Exception as e:
            kwargs = {}
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def register_custom_values(self, **kwargs):

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
