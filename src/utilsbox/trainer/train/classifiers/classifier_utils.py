import importlib
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from torchvision.transforms.functional import resize
from diffusers.utils.outputs import BaseOutput

from ...modeling_utils import ModelMixin, get_model_builder
from ....configuration import extract_init_dict


class BaseImageModel(ModelMixin):

    def __init__(self, resolution: int) -> None:
        super().__init__()

        self._resolution = resolution

    @property
    def resolution(self):
        return self._resolution

    def preprocess_images_input(self, images: torch.Tensor):

        if images.shape[-1] != self.resolution or images.shape[-2] != self.resolution:
            images = resize(images, (self.resolution, self.resolution), antialias=True)

    def __call__(self, images, *args, **kwargs):
        self.preprocess_images_input(images)
        return super(ModelMixin, self).__call__(images, *args, **kwargs)


@dataclass
class BaseImageClassifierOutput(BaseOutput):
    prediction: torch.Tensor
    feature: torch.Tensor


class BaseImageClassifier(BaseImageModel):

    WEIGHTS_NAME = 'classifier.bin'
    SAFETENSORS_WEIGHTS_NAME = 'classifier.safetensors'

    def __init__(
        self,
        resolution,
        num_classes,
        feature_dim,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(resolution, *args, **kwargs)
        self._num_classes = num_classes
        self._feature_dim = feature_dim

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def feature_dim(self):
        return self._feature_dim


def _reset_fc_impl(
    module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None
):
    """Reset the output class num of nn.Linear and return the input feature_dim of nn.Linear.

    Args:
        module (nn.Module): The specific model structure.
        reset_num_classes (int, optional): The new output class num. Defaults to None.
        visit_fc_fn (Callable, optional): Other operations to the nn.Linear of the input module. Defaults to None.

    Returns:
        feature_dim (int): The input feature_dim of nn.Linear.
    """

    if isinstance(module, nn.Sequential):

        if len(module) == 0:
            raise RuntimeError('fail to implement')

        if isinstance(module[-1], nn.Linear):
            feature_dim = module[-1].weight.shape[-1]

            if (
                reset_num_classes is not None
                and reset_num_classes != module[-1].weight.shape[0]
            ):
                module[-1] = nn.Linear(feature_dim, reset_num_classes)

            if visit_fc_fn is not None:
                visit_fc_fn(module[-1])

            return feature_dim
        else:
            return _reset_fc_impl(module[-1], reset_num_classes)

    children = list(module.named_children())
    if len(children) == 0:
        raise RuntimeError('Fail to implement')
    attr_name, child_module = children[-1]
    if isinstance(child_module, nn.Linear):
        feature_dim = child_module.weight.shape[-1]

        if (
            reset_num_classes is not None
            and reset_num_classes != child_module.weight.shape[0]
        ):
            setattr(module, attr_name, nn.Linear(feature_dim, reset_num_classes))

        if visit_fc_fn is not None:
            visit_fc_fn(getattr(module, attr_name))

        return feature_dim
    else:
        return _reset_fc_impl(child_module, reset_num_classes)


def reset_fc_layer(
    module: nn.Module, reset_num_classes: int = None, visit_fc_fn: Callable = None
) -> int:
    return _reset_fc_impl(module, reset_num_classes, visit_fc_fn)


class ExternalClassifier(BaseImageClassifier):

    def __init__(
        self,
        arch_name: str,
        pkg_name: str,
        num_classes: int,
        resolution=224,
        arch_kwargs={},
    ) -> None:

        _output_transform = None

        def _output_transform(m: nn.Linear):
            def hook_fn(module, input, output):
                return BaseImageClassifierOutput(prediction=output, feature=input[0])

            m.register_forward_hook(hook_fn)

        try:
            tv_module = importlib.import_module(pkg_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f'{pkg_name} module not found.')
        factory = getattr(tv_module, arch_name, None)
        if factory is None:
            raise RuntimeError(f'{pkg_name} do not support model {arch_name}')
        model = factory(**arch_kwargs)

        feature_dim = reset_fc_layer(model, num_classes, _output_transform)

        super().__init__(resolution, num_classes, feature_dim)

        self.model = model

    def forward(self, image: torch.Tensor, *args, **kwargs):
        return self.model(image)


class BaseClassifierWrapperModel(BaseImageClassifier):

    def __init__(
        self, inner_model_name: str, inner_model_kwargs: Optional[dict] = None, **kwargs
    ):

        if inner_model_kwargs is None:
            inner_model_kwargs = {}

        factory = get_model_builder(inner_model_name)
        inner_model = factory(**inner_model_kwargs)

        super(BaseClassifierWrapperModel, self).__init__(
            inner_model.resolution, inner_model.num_classes, inner_model.feature_dim
        )

        self.inner_model = inner_model

    @staticmethod
    def create_from_model_impl(cls, inner_model: ModelMixin, **kwargs):
        inner_model_name = inner_model.__class__.__name__
        # inner_model_kwargs = inner_model.extract_init_dict(inner_model._internal_dict)
        inner_model_kwargs = extract_init_dict(inner_model)
        inner_model_kwargs = inner_model_kwargs[0]
        # print(inner_model_name, inner_model_kwargs.keys())
        wrapped_model = cls(
            inner_model_name=inner_model_name,
            inner_model_kwargs=inner_model_kwargs,
            **kwargs,
        )
        wrapped_model.inner_model.load_state_dict(inner_model.state_dict())

        return wrapped_model

    @classmethod
    def create_from_model(cls, inner_model: ModelMixin, **kwargs):
        return cls.create_from_model_impl(cls, inner_model, **kwargs)
