# learn-llm-14days

[中文](README.md) | English

**Read through an entire LLM implementation in 14 days.**

An interactive study site that runs on your own machine. The textbook is [MiniMind](https://github.com/jingyaogong/minimind), a 100M-parameter Chinese LLM small enough to train from scratch on a laptop.
Each day you read one slice of **real source code** → run it right in the browser → **see the shape of every tensor, line by line** → change one thing and run again → answer a few questions.
From the tokenizer to Attention / MoE / KV cache, then Pretrain / SFT / LoRA / DPO / PPO / GRPO / Agent RL: in 14 days you walk the full model-and-training pipeline of a modern LLM.

No GPU needed. Every experiment finishes in seconds to a couple of minutes on a Mac (Apple Silicon) or an ordinary CPU.

> The lessons, quizzes and UI are written in Chinese. This page describes what the project is and how to run it.

![Home: the 14-day plan and your progress](docs/img/home.png)

## Who it is for

- You are fluent in Python and have heard of attention, RoPE, MoE, DPO — but you have **never written (or read end to end) an LLM implementation**.
- You want to know which lines of code a formula in a paper actually turns into, and how the shapes change step by step.
- You want to see how `input_ids / labels / loss_mask` are assembled in the training pipeline, and which lines make DPO, PPO and GRPO losses differ.

Not for you if: you want a conceptual introduction (we assume you know what softmax is), or you want to train a usable model (go to upstream MiniMind for that).

## Why MiniMind as the textbook

- The whole model, [`minimind/model/model_minimind.py`](minimind/model/model_minimind.py), is **288 lines**, yet it is a complete modern architecture: RMSNorm, RoPE + YaRN, GQA, QK-Norm, SwiGLU, MoE, KV cache, tied embeddings.
- The entire training pipeline lives in the same repo, one or two hundred lines per stage: Pretrain → SFT → LoRA / distillation → DPO / PPO / GRPO → Agent RL.
- The official weights are only 100M parameters (130MB in fp32); a forward pass takes under a second on CPU, so every concept can be **actually run** rather than illustrated.

## The 14 days

| Day | Topic | Sections | Labs | Quiz |
|---|---|---|---|---|
| **Part 1 · Model & inference**: the 288 lines of `model_minimind.py`, line by line | | | | |
| Day 1 | The big picture & the tokenizer | 12 | 12 | 20 |
| Day 2 | Skeleton: Config, Embedding, RMSNorm and the residual stream | 12 | 12 | 20 |
| Day 3 | Positional encoding: RoPE and YaRN | 11 | 11 | 22 |
| Day 4 | Attention: GQA, QK-Norm and the causal mask | 13 | 13 | 21 |
| Day 5 | FFN (SwiGLU), the Block, the full forward pass and the loss | 12 | 12 | 19 |
| Day 6 | MoE: routing, top-k, aux loss and active parameters | 13 | 13 | 23 |
| Day 7 | Inference: the generate loop, KV cache and sampling | 13 | 13 | 22 |
| **Part 2 · Data & training**: from jsonl to a falling loss | | | | |
| Day 8 | The data pipeline: from jsonl to (input_ids, labels) | 12 | 12 | 22 |
| Day 9 | The training loops of Pretrain and SFT | 13 | 13 | 22 |
| Day 10 | LoRA and knowledge distillation, hand-written (no peft) | 13 | 13 | 21 |
| **Part 3 · Alignment & agents** | | | | |
| Day 11 | DPO: squeezing a preference pair into one scalar loss | 13 | 12 | 23 |
| Day 12 | PPO: four models, GAE and the clipped objective | 13 | 13 | 21 |
| Day 13 | GRPO and the Rollout Engine | 12 | 12 | 21 |
| Day 14 | Agent RL, deployment and the final review | 14 | 14 | 28 |

62 chapters, 176 sections, 175 runnable labs and 305 quiz questions in total; 75–95 minutes a day. The full outline is in [PLAN.md](PLAN.md) (Chinese).

## How a day is structured

Like a class: **first the overview, then chapters by theme, one point per section, explain first and only then get hands-on.**

- **Overview** (first screen): where today's code sits in the whole model (a data-flow strip with today's protagonist highlighted), the chapters, and what you will be able to do afterwards.
- **Chapters** are distilled themes ("The model's configuration: MiniMindConfig"). **Sections** are one point per screen with declarative titles ("Where intermediate_size comes from").
- Every section follows the same rhythm:
  1. **Explanation** — what this code does, why it is written this way, how shapes change, where the pitfalls are, with source snippets inline;
  2. **▶ Verify it** — a micro-lab that targets exactly this point, one click to run;
  3. **🔧 Change one thing and rerun** — follow the prompt, edit one line; shapes that changed get highlighted;
  4. **Quiz** — multiple choice with full explanations;
  5. **🔑 One-sentence takeaway**.
- A few counter-intuitive spots use "guess first, then run": your choice is not revealed until the lab finishes and tells you whether you were right.
- Off-the-main-line details are folded away under "further reading"; dashed boxes mark optional sections. The last screen, "Today's recap", collects every section's takeaway and adds a few cross-chapter questions.
- The progress rail at the top is grouped by chapter, the sidebar shows the day's outline, `←` `→` turn pages, and every section has its own link. Progress is saved automatically.

![A Day 4 section: chapter lead-in → explanation with a source snippet → micro-lab](docs/img/section_top.png)

## Seeing tensor shapes: the heart of the site

`trace(...)` inside a lab annotates the source of a function with **which tensors each line wrote and what their shapes are**:

![Only the line relevant to this point is shown; a one-liner is unpacked into sub-expressions, each with its shape](docs/img/section.png)

- Colour chips are dimension symbols, consistent site-wide: `B` batch · `T` sequence length · `H` query heads · `KV` kv heads · `D` head_dim · `C` hidden · `I` FFN intermediate · `V` vocab · `E` experts; products like `B*T` and `KV*D` use gradients. How dimensions move through `view / transpose / reshape` is visible from the colours alone.
- A line that does several things at once (`xq.view(...).transpose(1, 2)`) is **unpacked** into sub-expressions, each annotated with its shape.
- Lines that did not execute are dimmed (you see at a glance which branch was taken); when a function is called several times, switch with `#1 #2 …`; click any chip for the tensor's details.
- **🧬 Variable trails**: one variable's shape evolution on a single line, e.g. `xq: [B,T,C] → [B,T,H,D] → [B,H,T,D]`.
- **🌲 Module call tree**: one row per submodule, "forward's arguments → return value", with parameter counts.
- Edit the code and run again: **tensors whose shape changed are boxed in orange, with the previous shape shown**.

The "📄 Source" tab lets you edit `model_minimind.py` and friends before running. The edit only applies to that run (it is injected through `sys.modules`); the files on disk are never touched, and you can "↺ Reset" or view the "⇄ Diff" at any time.

Besides the daily lessons there are two free-play pages:

- **🗺️ Model map**: set batch / sequence length / hidden size / layers / heads / MoE, run a real forward pass, and see three panels — whole model → one Block → inside a module — where every edge is a measured shape.
- **🧪 Playground**: write any code; `from learnkit import *` gives you all the visualisation tools.

![Model map](docs/img/map.png)

## Quick start

Requires Python 3.10+ and about 2GB of disk (1GB of dependencies + 0.8GB of weights and data).

```bash
git clone https://github.com/richardds5/learn-llm-14days.git
cd learn-llm-14days
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python scripts/download.py     # official weights ~660MB + 5 data files ~125MB
./start.sh                               # open http://127.0.0.1:8877
```

- Slow downloads from China: `venv/bin/python scripts/download.py --source modelscope` (needs `pip install modelscope`), or `export HF_ENDPOINT=https://hf-mirror.com` first.
- `venv/bin/python scripts/download.py --check` lists what is present and what is missing.
- Windows: use `venv\Scripts\python server.py` instead of `./start.sh` (lightly tested; feedback welcome).
- The server listens on 127.0.0.1 only. **It executes Python code sent from the web page** (that is the whole point), so never expose it to the internet.

Any lab can also be run from the terminal (visualisations degrade to text):

```bash
venv/bin/python runner.py lessons/day04/labs/qkv_proj.py
```

## Layout

```
learn-llm-14days/
├── start.sh  server.py  runner.py   # local server + the runner that executes labs in the minimind/ context
├── lib/learnkit/                    # tensor tracing & visualisation library (trace / show / heatmap / plot …); usable in your own scripts
├── web/                             # frontend: vanilla JS, no build step (Monaco / KaTeX / marked / highlight.js from CDN)
├── lessons/dayNN/                   # one directory per day: lesson.md (text) + quiz.json + labs/*.py
├── tools/check_lessons.py           # lesson linter: references / line numbers / quiz structure; --run executes all 175 labs
├── tools/make_plan.py  PLAN.md      # generates the outline from each day's frontmatter
├── tools/briefs/  AUTHORING.md      # authoring spec + the briefs used to have AI write and blind-review lessons
├── scripts/download.py              # downloads weights and data
└── minimind/                        # verbatim snapshot of upstream MiniMind, pinned to commit a3c7b01 (see below)
    ├── model/ trainer/ dataset/ scripts/ eval_llm.py
    ├── out/*.pth                    # downloaded weights (not in git)
    └── dataset/*.jsonl              # downloaded data (not in git)
```

Labs run with `minimind/` as the working directory, so references like `model/model_minimind.py:111` in the lessons and `from model.model_minimind import ...` in the labs are relative to it.
Your edited lab code is kept in `workspace/`, your progress in `progress.json` (delete it to reset).

## FAQ

**Port in use / `Address already in use`**: the previous server is still running (perhaps in another terminal tab). Find it with `lsof -i :8877` and `kill` it, or use another port: `./start.sh --port 9000`.

**A yellow banner says "server.py has been updated"**: the server code changed but was not restarted. `Ctrl+C` in the terminal that started it, then `./start.sh` again. Lessons, labs and frontend JS are read from disk live and do not need a restart.

**Studying against your own MiniMind checkout**: `MINIMIND_ROOT=/path/to/minimind ./start.sh`. The line references in the lessons were written against commit `a3c7b01`; after switching versions run `venv/bin/python tools/check_lessons.py` to see which references broke.

**A lab says the weights are missing**: `venv/bin/python scripts/download.py --check`.

**Running labs with a different interpreter**: `LEARN_PYTHON=/path/to/python ./start.sh`.

## Adding or editing lessons

All content is plain text:

- The format spec and the `learnkit` API are in [AUTHORING.md](AUTHORING.md) (Chinese): chapter/section format, hard requirements for labs, quiz structure, `focus` / `expand` options of `trace`.
- After editing run `venv/bin/python tools/check_lessons.py --run`: static checks on references, line numbers and quiz structure, then every lab is executed.
- [tools/briefs/](tools/briefs/) contains the briefs this repo used to have AI write and blind-review lessons; reuse them to add a day with the same workflow.

**How the content was made**: the lessons, quizzes and lab scripts were written by the author together with Claude — drafted day by day against the `AUTHORING.md` spec, blind-reviewed by a separate AI reviewer that re-checked every source reference and numeric claim, run end to end by `check_lessons.py`, and reworked over several rounds after the author studied each day himself. Mistakes may remain; please open an issue.

## Credits and license

- The textbook is [jingyaogong/minimind](https://github.com/jingyaogong/minimind) (Apache-2.0). The `minimind/` directory is a **verbatim snapshot** of its commit [`a3c7b01`](https://github.com/jingyaogong/minimind/commit/a3c7b01cc004d5de86aea961f20bf1e638e7c09e) (2026-09-10); not a character of source was changed (only images and the README were dropped). See [minimind/LICENSE](minimind/LICENSE) and [NOTICE](NOTICE). The version is pinned because the lessons cite it by line number.
- Weights come from [jingyaogong/minimind-3-pytorch](https://huggingface.co/jingyaogong/minimind-3-pytorch) and data from [jingyaogong/minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset); neither is included in this repository.
- Everything else (the site, the lessons, learnkit) is released under the [MIT](LICENSE) license.
