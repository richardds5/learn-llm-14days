# ---
# title: 工具结果回填之后，prompt 多出了哪些 token
# timeout: 60
# tasks:
#   - "把最后一条 {'role':'tool'} 复制成两条：模板只包一层 <|im_start|>user … <|im_end|>，还是包两层？回模板里 loop.first / messages[loop.index0-1].role != 'tool' 那两行找答案"
#   - "把 open_thinking 改成 True：彩带末尾的 '<think>\\n\\n</think>\\n\\n' 变成什么？这正是 rollout_single 里 thinking_ratio 控制的东西"
# ---
from learnkit import *

tok = get_tokenizer()
tools = [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式",
          "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}]

messages = [
    {"role": "system", "content": ""},                        # AgentRLDataset 里 tools 就挂在这条 system 上
    {"role": "user", "content": "算算 7109*2920"},
    {"role": "assistant", "content": '<tool_call>\n{"name": "calculate_math", "arguments": {"expression": "7109*2920"}}\n</tool_call>'},
    {"role": "tool", "content": '{"result": "20758280"}'},    # 👉 注意它被渲染成哪个 role
]
render = lambda n: tok.apply_chat_template(messages[:n], tokenize=False, add_generation_prompt=True,
                                           tools=tools, open_thinking=False)
ids = lambda n: tok(render(n), add_special_tokens=False)["input_ids"]

t1, t2 = ids(2), ids(4)      # 第 1 轮的 prompt（system+user） vs 工具结果回填后的第 2 轮 prompt
print(f"第 1 轮 prompt {len(t1)} tok → 第 2 轮 {len(t2)} tok；t2[:len(t1)] == t1 ? {t2[:len(t1)] == t1}")
print("👆 同一段历史的渲染结果严格前缀一致，所以 rollout_single 敢只在第 1 轮算一次 prompt_ids")

toks = tok.convert_ids_to_tokens(t2[len(t1) - 8:])            # 往前多取 8 个，看清衔接处
token_strip(toks, [1] * 8 + [0] * (len(toks) - 8),
            title="第 1 轮 prompt 尾部（1）→ 第 2 轮新增的 token（0）",
            legend={"1": "上一轮 prompt 已有", "0": "assistant 输出 + 工具结果 + 新的 generation prompt"})
