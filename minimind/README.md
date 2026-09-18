# minimind（快照）

这是 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 在
commit [`a3c7b01`](https://github.com/jingyaogong/minimind/commit/a3c7b01cc004d5de86aea961f20bf1e638e7c09e)（2026-09-10）的**原样快照**，
Apache-2.0 协议（见本目录 `LICENSE`）。

- 源文件一个字没改；去掉了上游的 `images/`、`README.md`、`README_en.md`、`CODE_OF_CONDUCT.md`。
- 课程里所有 `model/model_minimind.py:186-194` 这类引用都是按**这个版本**的行号写的，所以钉死在这个 commit，不要在这里 `git pull` 上游。
- `out/`（权重）和 `dataset/*.jsonl`（数据）不在 git 里，用仓库根目录的 `python scripts/download.py` 下载。
- 想对着别的版本学：把你自己 clone 的 minimind 路径设到环境变量 `MINIMIND_ROOT`，再跑 `tools/check_lessons.py` 看哪些行号引用失效。

完整文档、原理介绍、训练教程请看上游仓库。
