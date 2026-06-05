from dataclasses import dataclass
from typing import List


@dataclass
class ProductConfig:
    name: str          # "N86L" or "N86S"
    lines: List[str]   # ["1라인", "2라인", ...]
    seq_path: str      # path to seq txt file
    max_steps: int     # 53 or 56


@dataclass
class XGBConfig:
    num_seed: int = 42
    num_round: int = 1000
    early_stopping_rounds: int = 10
    objective: str = "reg:squarederror"
    eval_metric: str = "rmse"
    enable_categorical: bool = False
    max_depth: int = 6
    eta: float = 0.5  # matches legacy model default
    verbosity: int = 0
    tree_method: str = "hist"


@dataclass
class DataSettings:
    size_test: float = 0.2
    is_material_filtered: bool = True
    is_augment: bool = True
    is_splitlot: bool = True


PRODUCT_N86L = ProductConfig(
    name="N86L",
    lines=["1라인", "2라인", "3라인", "4라인", "5라인", "6라인"],
    seq_path="seq/seq_1_6.txt",
    max_steps=53,
)

PRODUCT_N86S = ProductConfig(
    name="N86S",
    lines=["7라인", "8라인", "9라인", "10라인"],
    seq_path="seq/seq_7_10.txt",
    max_steps=56,
)
