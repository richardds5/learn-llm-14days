# ---
# title: .pth → transformers 目录 → AutoModelForCausalLM 读回
# timeout: 180
# sources:
#   - scripts/convert_model.py
# tasks:
#   - "把 config.json 里的 auto_map 删掉再 from_pretrained：报什么错？这说明 register_for_auto_class 到底写了什么"
#   - "把 dtype 改成 torch.float16（脚本里的默认值）：maxdiff 变了吗？想想 out/full_sft_768.pth 本身是怎么存的（训练脚本写的是 torch.save({k: v.half().cpu()})）"
# ---
import json, os, shutil, tempfile, time, torch
from learnkit import *
import scripts.convert_model as cm
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoModelForCausalLM

# convert_model.py 把 lm_config 写在 `if __name__ == '__main__'` 里，import 进来时它是缺的，
# 而两个 convert_* 函数都直接读这个模块级全局 —— 手动补上。
cm.lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
out_dir = tempfile.mkdtemp(prefix="minimind_hf_")

t0, cwd = time.time(), os.getcwd()
os.chdir("scripts")                      # 函数里写死了 '../model/' 这种相对路径，只能站在 scripts/ 下调用
try:
    cm.convert_torch2transformers_minimind("../out/full_sft_768.pth", out_dir, dtype=torch.float32)
finally:
    os.chdir(cwd)
print(f"导出耗时 {time.time() - t0:.1f}s → {sorted(os.listdir(out_dir))}")

cfg = json.load(open(os.path.join(out_dir, "config.json"), encoding="utf-8"))
print("model_type =", cfg["model_type"])
print("auto_map   =", json.dumps(cfg["auto_map"], ensure_ascii=False))
print("👆 register_for_auto_class 干的就是这件事：把 model_minimind.py 拷进目录 + 写下「去哪个文件找哪个类」")

reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).eval()
print(f"AutoModelForCausalLM 读回的类是 {type(reloaded).__name__}，来自 {os.path.basename(type(reloaded).__module__)}")

original = MiniMindForCausalLM(cm.lm_config)
original.load_state_dict(torch.load("out/full_sft_768.pth", map_location="cpu"), strict=True)
torch.manual_seed(0)
input_ids = torch.randint(0, cm.lm_config.vocab_size, (3, 7))     # B=3, T=7
with torch.no_grad():
    a, b = original.eval()(input_ids).logits, reloaded(input_ids).logits
show(logits_torch=a, logits_transformers=b, dims=dict(B=3, T=7))
print("allclose(atol=1e-4) =", torch.allclose(a, b, atol=1e-4), " maxdiff =", (a - b).abs().max().item())
shutil.rmtree(out_dir, ignore_errors=True)
