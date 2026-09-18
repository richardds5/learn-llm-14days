# ---
# title: streamer.put 被调用了几次，每次收到多大一块
# timeout: 60
# tasks:
#   - "把 max_new_tokens 从 6 改成 9：表格会多几行？第 0 行的 shape 会变吗？"
#   - "把 model.generate(..., streamer=rec) 里的 streamer 去掉：表格会变成空的吗？（源码里三处 `if streamer:` 全部短路）"
# ---
import torch
from learnkit import *

model, tok = load_model("full_sft"), get_tokenizer()
text = tok.apply_chat_template([{"role": "user", "content": "用一句话介绍北京"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")["input_ids"]

rows = []
class Recorder:                      # TextStreamer 需要的接口只有 put / end 两个方法
    def put(self, value):            # generate 在循环前 put 一次、每步 put 一次
        rows.append([len(rows), "put", str(tuple(value.shape)), repr(tok.decode(value[0]))[1:-1][-28:]])
    def end(self):
        rows.append([len(rows), "end", "—", "—"])

model.generate(ids, max_new_tokens=6, do_sample=False, eos_token_id=None, streamer=Recorder())
table(rows, headers=["第几次回调", "方法", "收到的 tensor shape", "解码出来的内容（尾部）"],
      title=f"prompt 有 {ids.shape[1]} 个 token，max_new_tokens=6：put 一共被调用几次")
