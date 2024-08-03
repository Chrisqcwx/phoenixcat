import os
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models.inception import InceptionOutputs
from torchvision.transforms import (
    ToTensor,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomCrop,
)

from phoenixcat.models.classifiers import TorchvisionClassifier, BaseImageClassifier
from phoenixcat.configuration import register_to_pipeline_init, only_main_process
from phoenixcat.trainer.train_pipeline import TrainPipelineMixin
from phoenixcat.trainer.optimization import OptimizationConfig
from phoenixcat.trainer.losses import TorchLoss
from phoenixcat.logger import init_logger

from phoenixcat.data import DictAccumulator

output_dir = './test_output'
wandb_name = "example_train"
project_name = "example_train"

cifar_root = '<fill it>'

import logging

logger = logging.getLogger(__name__)

init_logger(os.path.join(output_dir, 'train.log'))


class ClassifierTrainPipeline(TrainPipelineMixin):
    """A simple classifier trainer"""

    _model_optimization_tag = 'model'
    _checkpoint_save_name = 'best_model'
    best_acc = 0

    @register_to_pipeline_init
    def __init__(
        self,
        output_dir: str,
        model: BaseImageClassifier,
        optimization_config: OptimizationConfig,
        validation_interval=1,
        save_train_status_interval=10,
        loss_fn='cross_entropy',
        seed: int = 0,
        accelerator: Accelerator | None = None,
    ) -> None:
        super().__init__(output_dir, seed, accelerator)
        self.model = model
        self.optimization_config = optimization_config
        self.loss_fn = TorchLoss(loss_fn)
        self.validation_interval = validation_interval
        self.save_train_status_interval = save_train_status_interval
        self.register_optimization(
            self._model_optimization_tag,
            self.model.parameters(),
            self.optimization_config,
        )

        self._checkpoint_save_folder = os.path.join(
            self.output_dir, self._checkpoint_save_name
        )
        os.makedirs(self._checkpoint_save_folder, exist_ok=True)

    @TrainPipelineMixin.register_execute_order(
        'epoch', interval='save_train_status_interval'
    )
    @TrainPipelineMixin.register_evaluate_function
    @torch.no_grad()
    def save_pretrained(self):
        return super().save_pretrained(self.output_dir)

    def forward_for_loss_and_acc(self, image, labels):
        pred = self.model(image).prediction
        acc = (pred.argmax(dim=-1) == labels).float().mean().item()
        return self.loss_fn(pred, labels), acc

    @TrainPipelineMixin.register_execute_main('train_step')
    def train_step(self, images, labels):

        loss, acc = self.forward_for_loss_and_acc(images, labels)

        self.optimization_manager.zero_grad(self._model_optimization_tag)

        self.backward(loss)
        self.optimization_manager.optimizer_step(self._model_optimization_tag)
        self.accelerator_log(
            {'train_loss': loss.item(), 'train_acc': acc},
        )

    @TrainPipelineMixin.register_execute_main('epoch')
    def train_loop(self):
        self.set_to_train_mode()
        train_dataloader = self.train_dataloader

        progress_bar = self.tqdm(train_dataloader, leave=False)
        for i, (images, labels) in enumerate(progress_bar):
            progress_bar.set_description(f'Epoch {self.execute_counts["epoch"]}')
            self.train_step(images, labels)

        self.optimization_manager.lr_scheduler_step(self._model_optimization_tag)

    @TrainPipelineMixin.register_execute_order('epoch', interval='validation_interval')
    @torch.no_grad()
    def validation(self):
        self.set_to_eval_mode()
        accumulator = DictAccumulator()
        val_dataloader = self.val_dataloader
        progress_bar = self.tqdm(val_dataloader, leave=False)

        for i, (image, labels) in enumerate(progress_bar):
            loss, acc = self.forward_for_loss_and_acc(image, labels)
            accumulator.add({'valid loss': loss.item(), 'valid acc': acc})

        result = accumulator.avg()
        self.accelerator_log(result)

        if result['valid acc'] > self.best_acc:
            self.best_acc = result['valid acc']
            if self.accelerator.is_main_process:
                self.unwrap_model(self.model).save_pretrained(
                    self._checkpoint_save_folder
                )

    def train_function(
        self, epoch_num, train_dataloader: DataLoader, val_dataloader: DataLoader
    ) -> None:

        self._max_epoch_num = epoch_num

        (
            self.model,
            self.train_dataloader,
            self.val_dataloader,
            self.optimization_manager,
        ) = self.accelerator_prepare(
            self.model,
            train_dataloader,
            val_dataloader,
            self.optimization_manager,
        )

        for epoch in range(epoch_num):
            self.train_loop()

    def set_to_train_mode(self):
        self.model.train()

    def set_to_eval_mode(self):
        self.model.eval()

    @property
    def is_end(self) -> bool:
        return self.execute_counts['epoch'] == self._max_epoch_num - 1


accelerator = Accelerator(log_with="wandb")

model = TorchvisionClassifier('resnet50', num_classes=100, resolution=32)


epoch_num = 100
validation_interval = 1
# save_checkpoint_interval = 20
save_train_status_interval = 20

trainset = CIFAR100(
    cifar_root,
    train=True,
    transform=Compose([RandomHorizontalFlip(), ToTensor()]),
)

testset = CIFAR100(cifar_root, train=False, transform=ToTensor())

batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

optimization_config = OptimizationConfig(
    'SGD', {'lr': 1e-2}, 'StepLR', {'step_size': 50, 'gamma': 0.25}
)

train_pipeline = ClassifierTrainPipeline(
    output_dir,
    model,
    optimization_config,
    validation_interval=validation_interval,
    # save_checkpoint_interval=save_checkpoint_interval,
    save_train_status_interval=save_train_status_interval,
    accelerator=accelerator,
)

accelerator.init_trackers(
    project_name=project_name,
    init_kwargs={
        "wandb": {
            "name": wandb_name,
            "dir": output_dir,
            "config": train_pipeline.config,
        }
    },
)

train_pipeline.train_function(epoch_num, trainloader, valloader)
