"""实验入口层：只做编排，不放核心逻辑。

规则：
    * 本包内的脚本只负责「读配置 -> 建 run -> 组装数据集/求解器/训练器 -> 调用 src -> 落盘」。
    * 任何算法实现都应放在 src/，便于被多个实验复用。
    * 每个 run 都会在 outputs/runs/ 下新建目录，历史 run 不被覆盖。

子目录：
    main        主实验
"""

from experiments.common import (
    build_dataset,
    make_parser,
    prepare_run,
    resolve_config,
    save_cmd,
)

__all__ = [
    "build_dataset",
    "make_parser",
    "prepare_run",
    "resolve_config",
    "save_cmd",
]
