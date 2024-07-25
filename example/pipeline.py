import os
import logging

import torch
import torchvision
from torch import nn, Tensor
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.configuration_utils import register_to_config

from phoenixcat.models import register_model
from phoenixcat.models.classifiers import BaseImageClassifier, BaseImageClassifierOutput
from phoenixcat.logger.logging import init_logger
from phoenixcat.random import seed_every_thing
from phoenixcat.trainer.trainer_utils import TrainingOutputFilesManager
from phoenixcat.configuration import PipelineMixin, register_to_pipeline_init

save_dir = './test_pipeline'

init_logger(os.path.join(save_dir, "pipeline.log"))

logger = logging.getLogger(__name__)


seed_every_thing(3)


@register_model
class VGG16_64(BaseImageClassifier):

    ignore_for_config = ['pretrained_weights']

    @register_to_config
    def __init__(self, num_classes, pretrained_weights=None):
        self.feat_dim = 512 * 2 * 2
        super(VGG16_64, self).__init__(64, num_classes, feature_dim=self.feat_dim)
        model = torchvision.models.vgg16_bn(weights=pretrained_weights)
        self.feature = model.features

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def forward(self, x: Tensor, *args, **kwargs):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        prediction = self.fc_layer(feature)

        return BaseImageClassifierOutput(prediction=prediction, feature=feature)


class TestPipeline(PipelineMixin):

    @register_to_pipeline_init
    def __init__(self, model, manager, scheduler, a_constant, no_seri):

        super().__init__()
        self.manager = manager

        self.register_save_values(init_constant=777)
        self.register_save_values(init_module=DDPMScheduler(num_train_timesteps=7))


scheduler = DDIMScheduler()

pipe = TestPipeline(
    model=VGG16_64(num_classes=10),
    manager=TrainingOutputFilesManager(),
    scheduler=scheduler,
    a_constant=["mao", 44],
    no_seri=nn.Identity(),
)


logger.info(f'------ test device and dtype ---------')
logger.info('origin')
logger.info(f'pipeline device={pipe.device}, dtype={pipe.dtype}')
logger.info(f'model device={pipe.model.device}, dtype={pipe.model.dtype}')

logger.info('to cuda and float16')
pipe.to(torch.device('cuda'), torch.float16)
logger.info(f'pipeline device={pipe.device}, dtype={pipe.dtype}')
logger.info(f'model device={pipe.model.device}, dtype={pipe.model.dtype}')

logger.info('to cpu and float32')
pipe.to(torch.device('cpu'), torch.float32)
logger.info(f'pipeline device={pipe.device}, dtype={pipe.dtype}')
logger.info(f'model device={pipe.model.device}, dtype={pipe.model.dtype}')

logger.info(f'------ test save and load ---------')

pipe.save_pretrained(save_dir)
pipe_new = TestPipeline.from_pretrained(save_dir)

logger.info('save pipeline')
logger.info(
    f'type={type(pipe)}, a_constant={pipe.a_constant}, init_constant={pipe.init_constant}'
)
logger.info(f'keys={list(pipe.__dict__.keys())}')

logger.info('load pipeline')
logger.info(
    f'type={type(pipe_new)}, a_constant={pipe_new.a_constant}, init_constant={pipe_new.init_constant}'
)
logger.info(f'keys={list(pipe_new.__dict__.keys())}')
