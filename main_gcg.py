# Standard library imports
import os
import json
from collections import namedtuple

import fire
import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress warnings from a specific library
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


Result = namedtuple('Result', 'prompt targetModel error', defaults=(None, None, None))


def run_gcg(target_model: str, 
            num_steps: int, 
            goal_prompt: str, 
            target: str,
            search_width: int = 64,
            topk: int = 64,
            seed: int = 42):

    model = AutoModelForCausalLM.from_pretrained(target_model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    messages = [{"role": "user", "content": goal_prompt}]

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=topk,
        seed=seed,
        verbosity="WARNING"
    )

    result = nanogcg.run(model, tokenizer, messages, target, config)

    return result.best_string


def run(target_model: str, num_steps: int, goal_prompt: str, result_path: str, target: str = "Sure, here is"):
    try:
        adv_suffix = run_gcg(target_model, num_steps, goal_prompt, target)
        adv_prompt = goal_prompt + " " + adv_suffix
        target_model_name = os.path.basename(target_model)
        result = [Result(prompt=adv_prompt,
                        targetModel=target_model_name
                        )._asdict()]
    except Exception as e:
        result = [Result(prompt=goal_prompt,
                        error=f"An error was detected during the GCG attack: {e}")._asdict()]
    with open(result_path, 'w', encoding="utf8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    fire.Fire(run)
    