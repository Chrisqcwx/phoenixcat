import os
import logging
import importlib
import functools

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
