# ---
# title: 同一份 state_dict 灌进 Qwen3ForCausalLM，logits 差多少
# timeout: 180
# sources:
#   - scripts/convert_model.py
# tasks:
#   - "把 common_config 里的 head_dim 改成 64 再 load_state_dict：在哪一行、报什么错？（q_proj 的 out_features 对不上）"
#   - "在 convert_model.py 里把 Qwen3Config/Qwen3ForCausalLM 换成 Qwen2Config/Qwen2ForCausalLM：strict=True 报的 missing key 和 unexpected key 各是哪一类？（一类是 Qwen2 多出来的，一类是 Qwen3 独有的）"
# ---
import json, os, shutil, tempfile, torch
from learnkit import *
import scripts.convert_model as cm
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoModelForCausalLM

cm.lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
out_dir = tempfile.mkdtemp(prefix="minimind_qwen_")
cwd = os.getcwd(); os.chdir("scripts")            # 函数里写死了 '../model/' 这种相对路径
try:
    cm.convert_torch2transformers("../out/full_sft_768.pth", out_dir, dtype=torch.float32)  # 👉 注意没有 _minimind 后缀
finally:
    os.chdir(cwd)

cfg = json.load(open(os.path.join(out_dir, "config.json"), encoding="utf-8"))
print("architectures =", cfg["architectures"], "| model_type =", cfg["model_type"], "| 有 auto_map 吗:", "auto_map" in cfg)
qwen = AutoModelForCausalLM.from_pretrained(out_dir).eval()        # 👉 不需要 trust_remote_code
print("读回的类 =", type(qwen).__name__, "| 目录里有 model_minimind.py 吗:", "model_minimind.py" in os.listdir(out_dir))

original = MiniMindForCausalLM(cm.lm_config)
original.load_state_dict(torch.load("out/full_sft_768.pth", map_location="cpu"), strict=True)
torch.manual_seed(0)
input_ids = torch.randint(0, cm.lm_config.vocab_size, (3, 7))     # B=3, T=7
with torch.no_grad():
    a, b = original.eval()(input_ids).logits, qwen(input_ids).logits
show(logits_minimind=a, logits_qwen3=b, dims=dict(B=3, T=7))
print("maxdiff =", (a - b).abs().max().item(), "| allclose(atol=0) =", torch.allclose(a, b, atol=0))
print("两边参数量:", sum(p.numel() for p in original.parameters()), sum(p.numel() for p in qwen.parameters()))
shutil.rmtree(out_dir, ignore_errors=True)
