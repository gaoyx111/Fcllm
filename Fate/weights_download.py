'''
import argparse
import glob
import os
from quantizer import quantize
from tqdm import tqdm
import re

def download_Qwen_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    # # from Hugging Face
    # folder = snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", allow_patterns="*.safetensor")
    # safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    weights_path = f"/root/workspace/Fate/model_weights/{model_name}/weights"
    safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    # quan_int8_path = os.path.join(quan_path, 'int8')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    # os.makedirs(quan_int8_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    expert_files = {}
    expert_int4_files = {}
    expert_int2_files = {}
    for layer in range(24):
        expert_files[layer] = {}
        expert_int4_files[layer] = {}
        expert_int2_files[layer] = {}
        for expert in range(8):
            expert_files[layer][expert] = {}
            expert_int4_files[layer][expert] = {}
            expert_int2_files[layer][expert] = {}

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)
    
        for name, param in tqdm(state.items(), leave=False):
            if "expert" not in name:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)
            else:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)

                # param_int8 = quantize(param, 8, 128)
                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)

                # param_int8_path = os.path.join(quan_int8_path, name)
                param_int4_path = os.path.join(quan_int4_path, name)
                param_int2_path = os.path.join(quan_int2_path, name)
                # torch.save(param_int8, param_int8_path)
                torch.save(param_int4, param_int4_path)
                torch.save(param_int2, param_int2_path)
                # torch.save(param_int3, param_int3_path)

def download_Qwen11_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    import torch

    # # from Hugging Face
    # folder = snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", allow_patterns="*.safetensor")
    # safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    weights_path = f"/root/workspace/Fate/model_weights/{model_name}/weights"
    safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    expert_files = {}
    expert_int4_files = {}
    expert_int2_files = {}
    for layer in range(24):
        expert_files[layer] = {}
        expert_int4_files[layer] = {}
        expert_int2_files[layer] = {}
        for expert in range(60):
            expert_files[layer][expert] = {}
            expert_int4_files[layer][expert] = {}
            expert_int2_files[layer][expert] = {}

    expert_pattern = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)_proj\.weight")

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)

        for name, param in tqdm(state.items(), leave=False):
            if "shared" in name or "expert" not in name:
                param_path = os.path.join(ori_path, name)
                save_file({"tensor": param}, param_path)
            else:
                match = expert_pattern.search(name)
                layer, expert_index, proj_type = match.groups()
                layer = int(layer)
                expert_index = int(expert_index)
                expert_files[layer][expert_index][proj_type] = param

                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)
                expert_int4_files[layer][expert_index][f'{proj_type}_nbits'] = param_int4.pop('nbits')
                expert_int4_files[layer][expert_index][f'{proj_type}_shape'] = param_int4.pop('shape')
                expert_int4_files[layer][expert_index][f'{proj_type}'] = param_int4.pop('W_q')
                expert_int4_files[layer][expert_index][f'{proj_type}_scale'] = param_int4.pop('scale')
                expert_int4_files[layer][expert_index][f'{proj_type}_zero'] = param_int4.pop('zero')

                expert_int2_files[layer][expert_index][f'{proj_type}_nbits'] = param_int2.pop('nbits')
                expert_int2_files[layer][expert_index][f'{proj_type}_shape'] = param_int2.pop('shape')
                expert_int2_files[layer][expert_index][f'{proj_type}'] = param_int2.pop('W_q')
                expert_int2_files[layer][expert_index][f'{proj_type}_scale'] = param_int2.pop('scale')
                expert_int2_files[layer][expert_index][f'{proj_type}_zero'] = param_int2.pop('zero')
    
    for layer_id, experts in expert_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(ori_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)
    for layer_id, experts in expert_int4_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(quan_int4_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)
    for layer_id, experts in expert_int2_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(quan_int2_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)

def download_Deepseek_weights_expert_3in1(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    # # from Hugging Face
    # folder = snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", allow_patterns="*.safetensor")
    # safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    weights_path = f"/root/weights/{model_name}/weights"
    safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = f"/root/weights"
    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    # quan_int8_path = os.path.join(quan_path, 'int8')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    # os.makedirs(quan_int8_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    expert_pattern = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)_proj\.weight")

    # with open('deepseekmoe.txt', 'w+') as txt_file:
    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)
    
        # for name, _ in tqdm(state.items(), leave=False):
        #     txt_file.write(f"{name}: {111}\n")
        experts_int4 = {}
        experts_int2 = {}

        for name, param in tqdm(state.items(), leave=False):
            if "expert" not in name:
                pass
                # param_path = os.path.join(ori_path, name)
                # torch.save(param, param_path)
            else:
                if "share" in name:
                    continue
                match = expert_pattern.search(name)
                layer, expert_index, proj_type = match.groups()

                # key = f"layer_{layer}_expert_{expert_index}"
                key = f"model.layers.{layer}.mlp.experts.{expert_index}.weight"
                # param_path = os.path.join(ori_path, name)
                # torch.save(param, param_path)

                # param_int8 = quantize(param, 8, 128)
                # param_int4 = quantize(param, 4, 64)
                # param_int2 = quantize(param, 2, 32)
                # param_int4 = quantize(param, 4)
                # param_int2 = quantize(param, 2)

                if key not in experts_int4:
                    experts_int4[key] = {}
                    experts_int2[key] = {}
                experts_int4[key][proj_type] = quantize(param, 4, 64)
                experts_int2[key][proj_type] = quantize(param, 2, 32)

        for key, expert_content in experts_int4.items():
            expert_path_file = os.path.join(quan_int4_path, key)
            torch.save(expert_content, expert_path_file)
            
        for key, expert_content in experts_int2.items():
            expert_path_file = os.path.join(quan_int2_path, key)
            torch.save(expert_content, expert_path_file)

                # name_int8 = name.replace("weight", "weight_int8")
                # name_int4 = name.replace("weight", "weight_int4")

                # param_int8_path = os.path.join(quan_int8_path, name)
                # param_int4_path = os.path.join(quan_int4_path, name)
                # param_int2_path = os.path.join(quan_int2_path, name)
                # # torch.save(param_int8, param_int8_path)
                # torch.save(param_int4, param_int4_path)
                # torch.save(param_int2, param_int2_path)

def download_Deepseek_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    # # from Hugging Face
    # folder = snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", allow_patterns="*.safetensor")
    # safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()

    # load from local
    weights_path = f"/root/weights/{model_name}/weights"
    safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = f"/root/weights"
    path = os.path.join(path, f"{model_name}")
    path = os.path.abspath(os.path.expanduser(path))
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    # quan_int8_path = os.path.join(quan_path, 'int8')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    # os.makedirs(quan_int8_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)

        for name, param in tqdm(state.items(), leave=False):
            if "expert" not in name:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)
            else:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)

                if "share" not in name:
                    # param_int8 = quantize(param, 8, 128)
                    param_int4 = quantize(param, 4, 64)
                    param_int2 = quantize(param, 2, 32)

                    # param_int8_path = os.path.join(quan_int8_path, name)
                    param_int4_path = os.path.join(quan_int4_path, name)
                    param_int2_path = os.path.join(quan_int2_path, name)
                    # # torch.save(param_int8, param_int8_path)
                    torch.save(param_int4, param_int4_path)
                    torch.save(param_int2, param_int2_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/workspace/Fate/model_weights")
    args = parser.parse_args()

    # download_JetMoE_weights_np("jetmoe/jetmoe-8b", args.path)

    # model = "Qwen/Qwen1.5-MoE-A2.7B"
    # download_Qwen11_weights(model, args.path)

    model = "deepseek-ai/deepseek-moe-16b-base"
    download_Deepseek_weights(model, args.path)

    # model = "jetmoe/jetmoe-8b"
    # download_JetMoE_weights(model, args.path)
'''

import argparse
import glob
import os
from quantizer import quantize
from tqdm import tqdm
import re

def download_Qwen_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    folder = snapshot_download("Qwen/Qwen1.5-MoE-A2.7B", allow_patterns="*.safetensor")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    # if "/" in model_name:
    #     model_name = model_name.split("/")[1].lower()

    # # 修改为当前目录的相对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # weights_path = os.path.join(current_dir, "model_weights", model_name, "weights")
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(current_dir, "model_weights", model_name)
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    # 后续逻辑完全保持原样
    expert_files = {}
    for layer in range(24):
        expert_files[layer] = {}
        for expert in range(8):
            expert_files[layer][expert] = {}

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)
        for name, param in tqdm(state.items(), leave=False):
            if "expert" not in name:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)
            else:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)

                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)

                param_int4_path = os.path.join(quan_int4_path, name)
                param_int2_path = os.path.join(quan_int2_path, name)
                torch.save(param_int4, param_int4_path)
                torch.save(param_int2, param_int2_path)

def download_Qwen11_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file
    import torch

    folder = snapshot_download(model_name, allow_patterns="*.safetensors")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    # if "/" in model_name:
    #     model_name = model_name.split("/")[1].lower()

    # # 路径修改
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # weights_path = os.path.join(current_dir, "model_weights", model_name, "weights")
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(current_dir, "model_weights", model_name)
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    # 后续逻辑完全保持原样
    expert_pattern = re.compile(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+)_proj\.weight")
    expert_files = {}
    expert_int4_files = {}
    expert_int2_files = {}
    for layer in range(24):
        expert_files[layer] = {}
        expert_int4_files[layer] = {}
        expert_int2_files[layer] = {}
        for expert in range(60):
            expert_files[layer][expert] = {}
            expert_int4_files[layer][expert] = {}
            expert_int2_files[layer][expert] = {}

    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)
        for name, param in tqdm(state.items(), leave=False):
            if "shared" in name or "expert" not in name:
                param_path = os.path.join(ori_path, name)
                save_file({"tensor": param}, param_path)
            else:
                match = expert_pattern.search(name)
                layer, expert_index, proj_type = match.groups()
                layer = int(layer)
                expert_index = int(expert_index)
                expert_files[layer][expert_index][proj_type] = param

                param_int4 = quantize(param, 4)
                param_int2 = quantize(param, 2)
                expert_int4_files[layer][expert_index][f'{proj_type}_nbits'] = param_int4.pop('nbits')
                expert_int4_files[layer][expert_index][f'{proj_type}_shape'] = param_int4.pop('shape')
                expert_int4_files[layer][expert_index][f'{proj_type}'] = param_int4.pop('W_q')
                expert_int4_files[layer][expert_index][f'{proj_type}_scale'] = param_int4.pop('scale')
                expert_int4_files[layer][expert_index][f'{proj_type}_zero'] = param_int4.pop('zero')

                expert_int2_files[layer][expert_index][f'{proj_type}_nbits'] = param_int2.pop('nbits')
                expert_int2_files[layer][expert_index][f'{proj_type}_shape'] = param_int2.pop('shape')
                expert_int2_files[layer][expert_index][f'{proj_type}'] = param_int2.pop('W_q')
                expert_int2_files[layer][expert_index][f'{proj_type}_scale'] = param_int2.pop('scale')
                expert_int2_files[layer][expert_index][f'{proj_type}_zero'] = param_int2.pop('zero')
    
    for layer_id, experts in expert_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(ori_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)
    for layer_id, experts in expert_int4_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(quan_int4_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)
    for layer_id, experts in expert_int2_files.items():
        for expert_id, expert_data in experts.items():
            expert_path = os.path.join(quan_int2_path, f"model.layers.{layer_id}.mlp.experts.{expert_id}.weight")
            save_file(expert_data, expert_path)

def download_Deepseek_weights(model_name, path):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    folder = snapshot_download(model_name, allow_patterns="*.safetensors")
    safetensor_files = glob.glob(os.path.join(folder, "*.safetensors"))

    # if "/" in model_name:
    #     model_name = model_name.split("/")[1].lower()

    # # 路径修改
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # weights_path = os.path.join(current_dir, "model_weights", model_name, "weights")
    # safetensor_files = glob.glob(os.path.join(weights_path, "*.safetensors"))

    path = os.path.join(current_dir, "model_weights", model_name)
    ori_path = os.path.join(path, 'original')
    quan_path = os.path.join(path, 'quantized')
    quan_int4_path = os.path.join(quan_path, 'int4')
    quan_int2_path = os.path.join(quan_path, 'int2')
    os.makedirs(ori_path, exist_ok=True)
    os.makedirs(quan_int4_path, exist_ok=True)
    os.makedirs(quan_int2_path, exist_ok=True)

    # 后续逻辑完全保持原样
    for safetensor_file in tqdm(safetensor_files, desc="Saving and quantizing"):
        state = load_file(safetensor_file)
        for name, param in tqdm(state.items(), leave=False):
            if "expert" not in name:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)
            else:
                param_path = os.path.join(ori_path, name)
                torch.save(param, param_path)

                if "share" not in name:
                    param_int4 = quantize(param, 4, 64)
                    param_int2 = quantize(param, 2, 32)

                    param_int4_path = os.path.join(quan_int4_path, name)
                    param_int2_path = os.path.join(quan_int2_path, name)
                    torch.save(param_int4, param_int4_path)
                    torch.save(param_int2, param_int2_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default=os.path.join(os.path.dirname(__file__), "model_weights"))
    args = parser.parse_args()

    # download_JetMoE_weights_np("jetmoe/jetmoe-8b", args.path)

    model = "Qwen/Qwen1.5-MoE-A2.7B"
    download_Qwen11_weights(model, args.path)

    model = "deepseek-ai/deepseek-moe-16b-base"
    download_Deepseek_weights(model, args.path)

    # model = "jetmoe/jetmoe-8b"
    # download_JetMoE_weights(model, args.path)