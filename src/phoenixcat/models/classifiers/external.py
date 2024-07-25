# Copyright 2024 Hongyao Yu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
