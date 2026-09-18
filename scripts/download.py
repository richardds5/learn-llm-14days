#!/usr/bin/env python3
"""MiniMind 学习站 - 权重与数据集下载器

用法:
  python scripts/download.py                       # 下载全部权重 + 数据集
  python scripts/download.py --weights              # 只下载权重
  python scripts/download.py --data                 # 只下载数据集
  python scripts/download.py --source modelscope     # 国内用户改用 ModelScope
  python scripts/download.py --check                 # 只检查文件是否齐全,不下载
"""
import argparse
import sys
import time
from pathlib import Path

# HuggingFace 仓库
HF_MODEL_REPO = "jingyaogong/minimind-3-pytorch"
HF_DATASET_REPO = "jingyaogong/minimind_dataset"
# ModelScope 仓库 (作者命名空间 gongjy)
MS_MODEL_REPO = "gongjy/minimind-3-pytorch"
MS_DATASET_REPO = "gongjy/minimind_dataset"

WEIGHTS = ["full_sft_768.pth", "pretrain_768.pth", "full_sft_768_moe.pth",
           "lora_identity_768.pth", "lora_medical_768.pth"]
DATA = ["lora_identity.jsonl", "dpo.jsonl", "lora_medical.jsonl",
        "agent_rl_math.jsonl", "rlaif.jsonl"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def human_size(n):
    if n <= 0:
        return "0KB"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n / 1024 / 1024:.1f}MB"

def download_one(name, kind, source, dest_dir):
    """下载单个文件, 返回 (是否成功, 文件大小, 耗时秒)"""
    target = dest_dir / name
    if target.exists() and target.stat().st_size > 0:
        print(f"  已存在,跳过: {name} ({human_size(target.stat().st_size)})")
        return True, target.stat().st_size, 0.0
    dest_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        if source == "hf":
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                print("缺少依赖 huggingface_hub, 请先执行: pip install huggingface_hub")
                sys.exit(1)
            repo_id = HF_MODEL_REPO if kind == "weights" else HF_DATASET_REPO
            repo_type = "model" if kind == "weights" else "dataset"
            hf_hub_download(repo_id=repo_id, filename=name, repo_type=repo_type,
                             local_dir=str(dest_dir))
        else:
            try:
                from modelscope.hub.file_download import model_file_download, dataset_file_download
            except ImportError:
                print("缺少依赖 modelscope, 请先执行: pip install modelscope")
                sys.exit(1)
            if kind == "weights":
                model_file_download(model_id=MS_MODEL_REPO, file_path=name, local_dir=str(dest_dir))
            else:
                dataset_file_download(dataset_id=MS_DATASET_REPO, file_path=name, local_dir=str(dest_dir))
    except SystemExit:
        raise
    except Exception as e:
        print(f"  下载失败: {name}: {e}")
        return False, 0, time.time() - t0

    elapsed = time.time() - t0
    size = target.stat().st_size if target.exists() else 0
    print(f"  {name}  {human_size(size)}  {elapsed:.1f}s")
    return True, size, elapsed

def run(kinds, source, dest, check_only):
    total_size, total_time, missing = 0, 0.0, 0
    for kind, names, sub in (("weights", WEIGHTS, "out"), ("data", DATA, "dataset")):
        if kind not in kinds:
            continue
        dest_dir = dest / sub
        print(f"\n[{'权重' if kind == 'weights' else '数据'}] -> {dest_dir}")
        for name in names:
            target = dest_dir / name
            if check_only:
                ok = target.exists() and target.stat().st_size > 0
                size = human_size(target.stat().st_size) if target.exists() else "-"
                print(f"  {'✓' if ok else '✗'} {name}  {size}")
                if not ok:
                    missing += 1
                continue
            ok, size, elapsed = download_one(name, kind, source, dest_dir)
            if ok:
                total_size += size
                total_time += elapsed
            else:
                missing += 1

    if check_only:
        if missing:
            print(f"\n缺失 {missing} 个文件")
            sys.exit(1)
        print("\n全部文件已就绪")
        return
    print(f"\n共 {human_size(total_size)}, 耗时 {total_time:.1f}s")
    if missing:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="MiniMind 学习站 - 下载权重与数据集",
        epilog="国内用户可设置环境变量 HF_ENDPOINT=https://hf-mirror.com 加速 HuggingFace 下载, 或直接使用 --source modelscope",
    )
    parser.add_argument("--weights", action="store_true", help="只下载权重文件")
    parser.add_argument("--data", action="store_true", help="只下载数据集文件")
    parser.add_argument("--source", choices=["hf", "modelscope"], default="hf",
                         help="下载源, 默认 huggingface, 国内用户可用 modelscope")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "minimind",
                         help="下载目标目录, 默认 <repo>/minimind")
    parser.add_argument("--check", action="store_true", help="只检查文件是否齐全 (不下载)")
    args = parser.parse_args()

    kinds = set()
    if args.weights:
        kinds.add("weights")
    if args.data:
        kinds.add("data")
    if not kinds:
        kinds = {"weights", "data"}

    run(kinds, args.source, args.dest.resolve(), args.check)

if __name__ == "__main__":
    main()
