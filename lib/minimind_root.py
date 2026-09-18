"""minimind 源码在哪。server.py / runner.py / learnkit 都从这里拿，只认一个规则：

    1. 环境变量 MINIMIND_ROOT（显式指定，比如想对着自己 clone 的 minimind 学）
    2. 本仓库自带的快照 ./minimind（默认，git clone 下来就有）
    3. 上级目录（把整个仓库放进 minimind 仓库里当 learn/ 子目录的旧布局）

判断标准就是那个目录下有没有 model/model_minimind.py。
"""
import os
from pathlib import Path

LEARN = Path(__file__).resolve().parents[1]  # lib/minimind_root.py → 仓库根目录


def find_minimind_root() -> Path:
    env = os.environ.get("MINIMIND_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if not (p / "model" / "model_minimind.py").is_file():
            raise FileNotFoundError(f"MINIMIND_ROOT={env} 下没有 model/model_minimind.py")
        return p
    for p in (LEARN / "minimind", LEARN.parent):
        if (p / "model" / "model_minimind.py").is_file(): return p.resolve()
    raise FileNotFoundError("找不到 minimind 源码：仓库里应该有 minimind/ 目录（git clone 自带），或者用环境变量 MINIMIND_ROOT 指定一个")
