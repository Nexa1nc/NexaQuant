import struct

def create_gguf_mock(filename, tensor_size_elements, n_tensors):
    with open(filename, 'wb') as f:
        # 1. GGUF Magic Header
        f.write(b'GGUF')
        # 2. Version (v3)
        f.write(struct.pack('<I', 3))
        # 3. Tensor Count & KV Count
        f.write(struct.pack('<QQ', n_tensors, 0))
        
        # 4. Write metadata for each mock tensor/layer
        for i in range(n_tensors):
            name = f"blk.{i}.attn_q".encode('utf-8')
            f.write(struct.pack('<Q', len(name)))
            f.write(name)
            f.write(struct.pack('<I', 1)) # 1 Dimension
            f.write(struct.pack('<Q', tensor_size_elements)) # Shape elements
            f.write(struct.pack('<I', 0)) # Type: FP32 (0)
            f.write(struct.pack('<Q', i * tensor_size_elements * 4)) # Data Offset
            
        # 5. Write mock weight data (zeros)
        total_data_bytes = tensor_size_elements * 4 * n_tensors
        f.write(b'\x00' * total_data_bytes)

# Creiamo 3 modelli di test con pesi reali simulati
create_gguf_mock('model_alpha.gguf', 1024 * 1024, 4) # Alpha: ~16MB (es. TinyLlama Mock)
create_gguf_mock('model_beta.gguf', 1024 * 1024, 8)  # Beta: ~32MB (es. Phi-3 Mock)
create_gguf_mock('model_gamma.gguf', 1024 * 1024, 12) # Gamma: ~48MB (es. Llama-3 Mock)

print("[SUITE] Created model_alpha.gguf (~16MB)")
print("[SUITE] Created model_beta.gguf (~32MB)")
print("[SUITE] Created model_gamma.gguf (~48MB)")
