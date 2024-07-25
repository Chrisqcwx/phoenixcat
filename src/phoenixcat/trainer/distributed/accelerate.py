import dataclasses
from diffusers.utils import is_accelerate_available

if is_accelerate_available():
    import accelerate

    accelerate.Accelerator()


@dataclasses.dataclass
class AcceleratorConfig:
    pass
