"""learnkit.models —— 快速拿到 MiniMind 模型 / tokenizer 的便捷函数。"""
import warnings

import torch

from .core import REPO_ROOT

_tokenizer = None


def build_model(seed=42, **config_kwargs):
    """随机初始化一个 MiniMind（eval 模式，CPU）。只看 shape 时建议 num_hidden_layers=2，秒级完成。
    例: build_model(num_hidden_layers=2, num_key_value_heads=2, use_moe=True)"""
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    torch.manual_seed(seed)
    return MiniMindForCausalLM(MiniMindConfig(**config_kwargs)).eval()


def weight_path(weight="full_sft", hidden_size=768, use_moe=False):
    return REPO_ROOT / "out" / f"{weight}_{hidden_size}{'_moe' if use_moe else ''}.pth"


def load_model(weight="full_sft", use_moe=False, device="cpu", dtype=torch.float32, **config_kwargs):
    """加载 out/ 下的官方权重。已下载: full_sft / pretrain / full_sft(use_moe=True)。"""
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    path = weight_path(weight, config_kwargs.get("hidden_size", 768), use_moe)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重 {path}。可用的权重: {[p.name for p in (REPO_ROOT / 'out').glob('*.pth')]}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MiniMindForCausalLM(MiniMindConfig(use_moe=use_moe, **config_kwargs))
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    return model.to(dtype).to(device).eval()


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(REPO_ROOT / "model"))
    return _tokenizer


def best_device():
    """M 系列芯片上返回 'mps'，否则 cuda / cpu。训练类实验用它。"""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
