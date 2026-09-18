"""learnkit —— MiniMind 学习站的 tensor 可视化小工具箱。

    from learnkit import *

    model = build_model(num_hidden_layers=2)
    with trace(model, fns=[Attention.forward], dims=dict(B=3, T=7)):
        model(torch.randint(0, 6400, (3, 7)))
"""
from .core import emit, IS_WEB, REPO_ROOT
from .tracer import trace, Trace
from .viz import show, heatmap, plot, live, bars, table, token_strip, note
from .models import build_model, load_model, get_tokenizer, weight_path, best_device

__all__ = ["trace", "show", "heatmap", "plot", "live", "bars", "table", "token_strip", "note",
           "build_model", "load_model", "get_tokenizer", "weight_path", "best_device", "REPO_ROOT", "IS_WEB"]
