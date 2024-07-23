import torchvision
from torch import nn, Tensor
from diffusers import DDIMScheduler
from diffusers.configuration_utils import register_to_config

from phoenixcat.models import register_model
from phoenixcat.models.classifiers import BaseImageClassifier, BaseImageClassifierOutput
from phoenixcat.logger.logging import init_logger
from phoenixcat.random import seed_every_thing

from phoenixcat.trainer.trainer_utils import TrainingOutputFilesManager
from phoenixcat.configuration import PipelineMixin, register_to_pipeline_init

init_logger("pipeline.log")


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
    def __init__(self, model, manager, scheduler, a_constant):

        super().__init__()
        self.manager = manager

        # self.register_modules(model=model, manager=manager, scheduler=scheduler)
        # self.register_constants(a_constant=a_constant)
        # self.register_to_config()


scheduler = DDIMScheduler()

pipe = TestPipeline(
    model=VGG16_64(num_classes=10),
    manager=TrainingOutputFilesManager(),
    scheduler=scheduler,
    a_constant=["mao", 44],
)

pipe.save_pretrained('./test_pipe')
pipe_new = TestPipeline.from_pretrained('./test_pipe')

print(type(pipe_new), pipe_new.a_constant)
