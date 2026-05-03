import struct

def create_gguf(filename):
    with open(filename, 'wb') as f:
        # 1. Magic
        f.write(b'GGUF')
        # 2. Version
        f.write(struct.pack('<I', 3))
        # 3. Tensor Count & KV Count
        f.write(struct.pack('<QQ', 1, 0))
        
        # 4. Tensor Info
        name = b"blk.0.attn_q"
        f.write(struct.pack('<Q', len(name)))
        f.write(name)
        f.write(struct.pack('<I', 1)) # n_dims
        f.write(struct.pack('<Q', 4096)) # ne[0]
        f.write(struct.pack('<I', 0)) # type (FP32)
        f.write(struct.pack('<Q', 0)) # offset
        
        # 5. Data (Aggiungiamo un po' di dati dummy dopo l'header)
        f.write(b'\x00' * 4096 * 4)

create_gguf('test_model.gguf')
print("Modello di test creato: test_model.gguf")
