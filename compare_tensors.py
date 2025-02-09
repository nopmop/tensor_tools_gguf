import argparse
import torch
from gguf import GGUFReader

def load_source_model(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)

def map_tensor_name(name):
    if name.startswith('bert.'):
        name = name[len('bert.'):]
    
    tensor_mappings = {
        'embeddings.word_embeddings.weight': 'token_embd.weight',
        'embeddings.position_embeddings.weight': 'position_embd.weight',
        'embeddings.token_type_embeddings.weight': 'token_types.weight',
        'embeddings.LayerNorm.weight': 'token_embd_norm.weight',
        'embeddings.LayerNorm.bias': 'token_embd_norm.bias',
        'embeddings.position_ids': 'position_ids',
        'cls.predictions.bias': 'cls.predictions.bias',
        'cls.predictions.transform.LayerNorm.bias': 'cls.predictions.transform.layer_norm.bias',
        'cls.predictions.transform.LayerNorm.weight': 'cls.predictions.transform.layer_norm.weight',
        'cls.predictions.transform.dense.bias': 'cls.predictions.transform.dense.bias',
        'cls.predictions.transform.dense.weight': 'cls.predictions.transform.dense.weight',
        'pooler.dense.weight': 'pooler.dense.weight',
        'pooler.dense.bias': 'pooler.dense.bias',
        'cls.predictions.decoder.weight': 'cls.decoder.weight',
        'cls.predictions.decoder.bias': 'cls.decoder.bias',
    }

    for i in range(12):
        tensor_mappings.update({
            f'encoder.layer.{i}.attention.output.LayerNorm.bias': f'blk.{i}.attn_output_norm.bias',
            f'encoder.layer.{i}.attention.output.LayerNorm.weight': f'blk.{i}.attn_output_norm.weight',
            f'encoder.layer.{i}.attention.output.dense.bias': f'blk.{i}.attn_output.bias',
            f'encoder.layer.{i}.attention.output.dense.weight': f'blk.{i}.attn_output.weight',
            f'encoder.layer.{i}.attention.self.key.bias': f'blk.{i}.attn_k.bias',
            f'encoder.layer.{i}.attention.self.key.weight': f'blk.{i}.attn_k.weight',
            f'encoder.layer.{i}.attention.self.query.bias': f'blk.{i}.attn_q.bias',
            f'encoder.layer.{i}.attention.self.query.weight': f'blk.{i}.attn_q.weight',
            f'encoder.layer.{i}.attention.self.value.bias': f'blk.{i}.attn_v.bias',
            f'encoder.layer.{i}.attention.self.value.weight': f'blk.{i}.attn_v.weight',
            f'encoder.layer.{i}.intermediate.dense.bias': f'blk.{i}.ffn_up.bias',
            f'encoder.layer.{i}.intermediate.dense.weight': f'blk.{i}.ffn_up.weight',
            f'encoder.layer.{i}.output.LayerNorm.bias': f'blk.{i}.layer_output_norm.bias',
            f'encoder.layer.{i}.output.LayerNorm.weight': f'blk.{i}.layer_output_norm.weight',
            f'encoder.layer.{i}.output.dense.bias': f'blk.{i}.ffn_down.bias',
            f'encoder.layer.{i}.output.dense.weight': f'blk.{i}.ffn_down.weight',
        })

    if name in tensor_mappings:
        return tensor_mappings[name]
    print(f"Warning: Cannot map tensor {name!r}")
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare tensors between a PyTorch model and a GGUF model")
    parser.add_argument('--source_model', type=str, required=True, help="Path to the source PyTorch model")
    parser.add_argument('--gguf_model', type=str, required=True, help="Path to the GGUF model")
    args = parser.parse_args()

    # Load source model
    source_model = load_source_model(args.source_model)

    # Load GGUF model using GGUFReader
    reader = GGUFReader(args.gguf_model)
    gguf_tensors = {tensor[0]: tensor[1] for tensor in reader.tensors}

    # Log the number of tensors from source model
    source_tensors = list(source_model.keys())

    # Log the number of tensors from GGUF model
    gguf_tensor_names = list(gguf_tensors.keys())

    # Map and print the mapping between source and GGUF tensors
    source_to_gguf_mapping = {}
    for name in source_tensors:
        mapped_name = map_tensor_name(name)
        source_to_gguf_mapping[name] = mapped_name
        print(f"{name} -> {mapped_name}")

    # Identify missing tensors
    mapped_source_tensors = set(source_to_gguf_mapping.values())
    missing_tensors = mapped_source_tensors - set(gguf_tensor_names)
    extra_tensors = set(gguf_tensor_names) - mapped_source_tensors

    print(f"Number of source tensors: {len(source_tensors)}")
    print(f"Number of GGUF tensors: {len(gguf_tensor_names)}")
    print(f"Missing tensors in GGUF model: {missing_tensors}")
    print(f"Extra tensors in GGUF model: {extra_tensors}")

    if len(missing_tensors) > 0 or len(extra_tensors) > 0:
        exit()

    # Ensure that all source tensors are correctly mapped
    assert not missing_tensors, f"Missing tensors: {missing_tensors}"

    # Example conversion logic
    for name, param in source_model.items():
        new_name = source_to_gguf_mapping[name]
        if new_name and new_name in gguf_tensors:
            gguf_tensors[new_name] = param

    # Save the updated GGUF model
    reader.save('updated_gguf_model.gguf')

if __name__ == "__main__":
    main()
