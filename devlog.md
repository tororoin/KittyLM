# Pretraining LLMs Software Project -

---
*Devlog 2024.10.31* - Brainstorming

- Architecture : GPT-2 [should we train a smaller variant?]
-- Derive scaling laws after getting datasset.
-- Test Positional Embeddings with LearnedPE, RotaryPE for longer ctx (and possibly just switch to RoPE permanently)
-- Test Attention variants, starting with MHA -> GQA -> MQA -> MLKV(should we try this?)

- Data
-- Look for datasets that is a mixture of general text, high quality scientific article, code, math, etc [objective is to understand what kinda data is beneficial for a good llm] and train model on this. Also train model on only general text to see what the attention heads learn and what information they store i.e. MechInterp?
-- anything else?

- Targets / Evaluations
**Okay so I have not given this much thought as I am not really familiar with this area, but what could be some possible ways / tasks to evaluate our model? I cannot really think of any tasks. Maybe Michael can suggest some?**

- Analysis
-- Look at what each attention head learns, what information it stores (im a bit new to MechInterp but I'll get a fair idea after reading some papers)

- Evaluate on
-- RelPron
-- mcqgetting

 no lora, only a classification head

---

*Devlog 2024.11.03* - From the discussion, I feel we should take up evaluation tasks from papers like gpt / llama stuff that i've looked at

- gsm8k : grade school math problems by OpenAI ig
- mmlu
-

I'm writing the rest of the gpt model along with the training code.  Michael wants most of it by Thursday, but there is no pressure