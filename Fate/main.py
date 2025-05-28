import argparse
from utils import str2bool
import os 
from transformers import AutoTokenizer
import time
import torch
from dataset import get_ChatGPT_prompts_inputs, get_gsm8k_inputs, get_openai_humaneval_inputs
from tqdm import tqdm
import torch.profiler
from models.Qwen.modeling_qwen_moe import Qwen2MoeForCausalLM
from models.DeepSeekMoE.modeling_deepseek import DeepseekForCausalLM

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-MoE-A2.7B", help="The model name.")
    # parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-moe-16b-base", help="The model name.")
    # parser.add_argument("--path", type=str, default="/root/workspace/Fate/model_weights", help="The path to the model weights.")
    parser.add_argument("--path", type=str, default= os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights"), help="The path to the model weights.")
    parser.add_argument("--early_stopping", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--memory_budget", type=int, default=0, help="GB")
    # parser.add_argument("--device", type=str, default='cuda:1')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--overlap", type=str2bool, nargs='?', const=True, default=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()
        
    model_name = args.model
    # model_name = "deepseek-ai/deepseek-moe-16b-base"

    if model_name == "Qwen/Qwen1.5-MoE-A2.7B":
        # tokenizer = AutoTokenizer.from_pretrained("/root/workspace/Fate/model_weights/qwen1.5-moe-a2.7b/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights", "Qwen", "Qwen1.5-MoE-A2.7B", "tokenizer"))
        model = Qwen2MoeForCausalLM(args)
    elif model_name == "deepseek-ai/deepseek-moe-16b-base":
        # args.path = "/root/weights"
        # tokenizer = AutoTokenizer.from_pretrained("/root/weights/deepseek-moe-16b-base/tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base", cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights", "deepseek-ai", "deepseek-moe-16b-base", "tokenizer"))
        model = DeepseekForCausalLM(args)
    
    model.eval()

    input_prompt = "Hey, are you conscious? Can you talk to me?"
    input_tokenizer = tokenizer(input_prompt, return_tensors="pt")
    input_ids = input_tokenizer.input_ids.to(args.device)
    attention_mask = input_tokenizer.attention_mask.to(args.device)

    # warmup
    for i in range(1):
        output_ids = model.generate(input_ids)

    start = time.time()
    (output_ids, prefill_time) = model.generate(input_ids, attention_mask=attention_mask, expriment_mode="decoding")
    end = time.time()
    print("latency = " + str(end-start))

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(outputs)