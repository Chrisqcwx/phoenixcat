import os
from dataclasses import dataclass
from typing import Optional

import sys

sys.path.append("../src")

from phoenixcat.auto import config_dataclass_wrapper


@config_dataclass_wrapper(config_name="config.json")
@dataclass
class TeacherConfig:
    name: str
    classname: Optional[str] = None


@config_dataclass_wrapper(config_name="config.json")
@dataclass
class SchoolConfig:
    name: str
    teacher: Optional[TeacherConfig] = None


@config_dataclass_wrapper(config_name="config.json")
@dataclass
class PersonConfig:
    name: str
    age: int
    school: SchoolConfig | None


teacher_config = TeacherConfig(name="Tom", classname="Math")
school_config = SchoolConfig(name="ABC", teacher=teacher_config)
config = PersonConfig(name="Tom", age=18, school=school_config)

print(config)

save_dir = './test_config'

config.save_pretrained(save_dir)

new_config: PersonConfig = PersonConfig.from_pretrained(save_dir)
print(new_config)
