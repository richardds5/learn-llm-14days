# ---
# title: 同一个基座，热插拔两份官方 LoRA 权重
# timeout: 120
# tasks:
#   - "把第二个 load_lora 换回 lora_identity_768.pth，再问一次医学问题：两份 LoRA 能同时生效吗？"
#   - "在 load_lora 之前先 `for _, m in base.named_modules():` 把所有 `m.lora.B.weight.data.zero_()`，再问一次：回答是不是退回基座的样子？"
# ---
from learnkit import load_model, get_tokenizer
from model.model_lora import apply_lora, load_lora

tok = get_tokenizer()


def chat(model, prompt, max_new_tokens=48):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    out = model.generate(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
                        max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


Q_ID, Q_MED = "你叫什么", "头发稀少细软可以植发吗"
base = load_model("full_sft")
print(f"[base]           {Q_ID} -> {chat(base, Q_ID)!r}")
print(f"[base]           {Q_MED} -> {chat(base, Q_MED)!r}")

apply_lora(base)                                # 挂上空壳（ΔW=0，此刻输出还和上面一样）
load_lora(base, "out/lora_identity_768.pth")    # 👉 填入自我认知 LoRA
print(f"[+lora_identity] {Q_ID} -> {chat(base, Q_ID)!r}")

load_lora(base, "out/lora_medical_768.pth")     # 👉 直接覆盖，不用重新 apply_lora
print(f"[+lora_medical]  {Q_MED} -> {chat(base, Q_MED)!r}")
print(f"[+lora_medical]  {Q_ID} -> {chat(base, Q_ID)!r}")
