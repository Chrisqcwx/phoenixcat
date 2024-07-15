from diffusers.configuration_utils import register_to_config

from .classifier_utils import ExternalClassifier
from ..modeling_utils import register_model


@register_model
class TorchvisionClassifier(ExternalClassifier):

    ignore_for_config = ['weights']

    @register_to_config
    def __init__(
        self,
        arch_name: str,
        num_classes: int,
        resolution=224,
        weights=None,
        arch_kwargs={},
    ) -> None:

        if weights is not None:
            arch_kwargs['weights'] = weights

        super().__init__(
            arch_name=arch_name,
            pkg_name='torchvision.models',
            num_classes=num_classes,
            resolution=resolution,
            arch_kwargs=arch_kwargs,
        )


@register_model
class ResNeStClassifier(ExternalClassifier):

    ignore_for_config = ['weights']

    @register_to_config
    def __init__(
        self,
        arch_name: str,
        num_classes: int,
        resolution=224,
        weights=None,
        arch_kwargs={},
    ) -> None:

        if weights is not None:
            arch_kwargs['weights'] = weights

        super().__init__(
            arch_name=arch_name,
            pkg_name='resnest.torch',
            num_classes=num_classes,
            resolution=resolution,
            arch_kwargs=arch_kwargs,
        )
