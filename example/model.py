import torchvision
from torch import nn, Tensor
from diffusers.configuration_utils import register_to_config

from phoenixcat.models import register_model
from phoenixcat.models.classifiers import BaseImageClassifier, BaseImageClassifierOutput
from phoenixcat.download.hugging_face.api_download import download_from_huggingface
from phoenixcat.logger.logging import init_logger
from phoenixcat.random import seed_every_thing


init_logger("model.log")


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


model = VGG16_64(num_classes=127, pretrained_weights='DEFAULT')
model.save_pretrained('./test')

from phoenixcat.models import auto_model_from_pretrained

model_new = auto_model_from_pretrained('./test')
print(type(model_new))
