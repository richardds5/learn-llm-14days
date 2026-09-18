"""在 minimind 源码目录的上下文里执行一段实验代码（cwd = minimind/，所以 from model.model_minimind import ... 直接可用）。

    venv/bin/python runner.py lessons/day04/labs/attn_trace.py
    venv/bin/python runner.py code.py --overrides overrides.json

overrides.json: {"model/model_minimind.py": "/abs/path/to/modified.py", ...}
网页里「改源码」就是靠它：把修改后的文件当成同名模块塞进 sys.modules，minimind 原文件一个字都不会动。
"""
import os
import sys
import json
import runpy
import warnings
import importlib
import importlib.util
from pathlib import Path

LEARN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LEARN_DIR / "lib"))
from minimind_root import find_minimind_root  # noqa: E402

REPO_ROOT = find_minimind_root()


def install_overrides(mapping: dict):
    for rel, src in mapping.items():
        mod_name = rel[:-3].replace("/", ".")
        pkg, _, leaf = mod_name.rpartition(".")
        spec = importlib.util.spec_from_file_location(mod_name, src)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        if pkg: setattr(importlib.import_module(pkg), leaf, module)
        print(f"[runner] 使用你修改过的 {rel}", flush=True)


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    code_path = Path(sys.argv[1]).resolve()
    overrides = {}
    if "--overrides" in sys.argv:
        overrides = json.loads(Path(sys.argv[sys.argv.index("--overrides") + 1]).read_text(encoding="utf-8"))

    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    warnings.filterwarnings("ignore")
    sys.argv = [str(code_path)]

    os.environ["LEARNKIT_OVERRIDES"] = json.dumps(overrides)
    if overrides: install_overrides(overrides)
    if not os.environ.get("LEARNKIT_EVENTS"):
        runpy.run_path(str(code_path), run_name="__main__")
        return

    # 网页模式：print 也走事件通道，这样文字输出和图表能像 notebook 一样按先后顺序交错显示
    from learnkit.core import emit

    class EventStdout:
        encoding = "utf-8"
        def write(self, text):
            if text: emit({"type": "stdout", "text": text})
            return len(text)
        def flush(self): pass
        def isatty(self): return False

    sys.stdout = EventStdout()
    try:
        runpy.run_path(str(code_path), run_name="__main__")
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        tb = traceback.extract_tb(e.__traceback__)
        keep = [f for f in tb if "runpy" not in f.filename and f.filename != __file__]
        text = "Traceback (most recent call last):\n" + "".join(traceback.format_list(keep or tb)) + "".join(traceback.format_exception_only(type(e), e))
        line = next((f.lineno for f in reversed(keep) if f.filename == str(code_path)), None)
        emit({"type": "error", "traceback": text.replace(str(code_path), "实验脚本"), "line": line, "name": type(e).__name__})
        sys.exit(1)


if __name__ == "__main__":
    main()
