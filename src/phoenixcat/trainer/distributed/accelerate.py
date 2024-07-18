import dataclasses

import accelerate

accelerate.Accelerator()

@dataclasses.dataclass
class AcceleratorConfig:
    pass