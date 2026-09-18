# ---
# title: kwargs.get 的覆盖顺序：父类默认值 vs MiniMind 默认值
# timeout: 60
# tasks:
#   - "在 FIELDS 里加一行 ('rope_theta', 1234)：PretrainedConfig 认识这个字段吗？第 2 列会显示什么？"
#   - "把 model/model_minimind.py L19 的 kwargs.get('bos_token_id', 1) 整行删掉再跑：第 3 列变成什么？（这一行就是「后写覆盖前写」本身）"
# sources:
#   - model/model_minimind.py
# ---
from learnkit import *
from transformers import PretrainedConfig
from model.model_minimind import MiniMindConfig

MISSING = "<父类没有这个属性>"
FIELDS = [("bos_token_id", 99), ("eos_token_id", 99), ("tie_word_embeddings", False), ("vocab_size", 111)]

rows = []
for name, override in FIELDS:
    parent = getattr(PretrainedConfig(), name, MISSING)          # super().__init__(**kwargs) 之后的样子
    mine = getattr(MiniMindConfig(), name)                        # 再被 kwargs.get(name, 默认值) 覆盖一次
    passed = getattr(MiniMindConfig(**{name: override}), name)    # 调用方显式传值时还能不能透传
    rows.append([name, repr(parent), repr(mine), f"{name}={override!r} → {passed!r}"])

table(rows, headers=["字段", "PretrainedConfig() 裸跑", "MiniMindConfig() 最终值", "显式传值时"],
      title="MiniMindConfig.__init__ 里的 kwargs.get 把父类默认值覆盖掉了")
