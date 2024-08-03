import os
import argparse
import logging
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
from phoenixcat.configuration import register_to_pipeline_init, auto_create_cls
from phoenixcat.trainer.train_pipeline import TrainPipelineMixin
from phoenixcat.trainer.optimization import OptimizationConfig
from phoenixcat.trainer.losses import TorchLoss
from phoenixcat.trainer.data import getDataLoader
from phoenixcat.logger import init_logger
from phoenixcat.parser import ConfigParser
from phoenixcat.data import DictAccumulator

logger = logging.getLogger(__name__)


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


def arg_parse():

    parser = argparse.ArgumentParser(description='train classifier')
    parser.add_argument('--config', '-c', type=str, default='config.yaml')
    parser.add_argument('--output', '-o', type=str, default='./output')
    parser.add_argument('--cuda', '-g', type=str, default=None)
    return parser.parse_args()


class Cifar100ClassifierConfigParser(ConfigParser):

    def __init__(self, config, config_path: str | None = None):
        super().__init__(config, config_path)

        self.output_dir = self.get('output_dir', './.output')

        init_logger(os.path.join(self.output_dir, 'train.log'))

    def create_dataset(self, train=True, transforms=None):
        return self.create_from_name(
            'torchvision.datasets.CIFAR100',
            kwargs={
                'root': self.get('dataset_path'),
                'train': train,
                'download': True,
                'transform': transforms,
            },
        )

    def create_pipeline(self):
        model = self.create_model()
        optimization_config = self.create_optimization_config()
        try:
            accelerator = self.create_accelerator()
            logger.info(f'use accelerator')

            project_name = self.get('project_name', 'train_classifier')
            logger.info(f'init trackers with project name {project_name}')
            init_kwargs = {
                "wandb": {
                    "name": self.get("wandb_name", project_name),
                    "dir": self.output_dir,
                    "config": self.config,
                }
            }
            accelerator.init_trackers(
                project_name=project_name,
                init_kwargs=init_kwargs,
            )

        except Exception as e:
            accelerator = None
            logger.info(f'no accelerator')

        train_pipeline = auto_create_cls(
            ClassifierTrainPipeline,
            config['pipeline'],
            output_dir=self.output_dir,
            model=model,
            optimization_config=optimization_config,
            accelerator=accelerator,
        )

        return train_pipeline

    def run(self):
        train_pipeline = self.create_pipeline()

        trainset = self.create_dataset(
            train=True, transforms=self.create_transform("train_transforms")
        )
        valset = self.create_dataset(
            train=False, transforms=self.create_transform("val_transforms")
        )

        batch_size = self.get('train_batch_size')
        test_batch_size = self.get('test_batch_size', batch_size)
        num_workers = self.get('num_workers', 4)
        train_loader = getDataLoader(
            trainset,
            batch_size=batch_size,
            config={
                'num_workers': num_workers,
            },
        )
        val_loader = getDataLoader(
            valset,
            batch_size=test_batch_size,
            config={
                'num_workers': num_workers,
            },
        )

        epoch_num = self.get('epochs')

        if train_pipeline.is_main_process:
            self.save_config(os.path.join(self.output_dir, 'config.yaml'))
        train_pipeline.train_function(epoch_num, train_loader, val_loader)


if __name__ == '__main__':
    args = arg_parse()
    # print(args.cuda, type(args.cuda), args.cuda is None)
    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    config = Cifar100ClassifierConfigParser.from_config_file(args.config)
    config.run()
