#!/usr/bin/env python3
"""
NEXAQUANT v3.0 - Character-level Text Predictor Training & Generation Script
(C) 2026 Nexa1nc - AGPL v3 Open Source AI Democracy
"""
import os
import sys
import subprocess
import random

# Color constants for premium CLI look
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"

SHAKESPEARE_SAMPLE = """
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
the heart ache and the thousand natural shocks
that flesh is heir to tis a consummation
devoutly to be wishd to die to sleep
to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come
when we have shuffled off this mortal coil
must give us pause there is the respect
that makes calamity of so long life
"""

def print_header():
    print(CYAN + """
    ========================================================================
     🚀 NEXAQUANT v3.0: Ultra-Low RAM Character-Level Text Generator Trainer
    ========================================================================
     Democratic AI. Train next-character prediction models natively on your PC
     using high-performance 1.58-bit Stochastic Integer Accumulators!
    ========================================================================
    """ + RESET)

def get_or_create_input_text():
    if os.path.exists("input.txt"):
        print(f"[INFO] Found existing '{YELLOW}input.txt{RESET}'. Reading dataset...")
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print(f"[INFO] No '{YELLOW}input.txt{RESET}' found. Creating a beautiful Shakespeare sample to train on...")
        text = SHAKESPEARE_SAMPLE
        with open("input.txt", "w", encoding="utf-8") as f:
            f.write(text.strip())
    
    # Clean the text to keep it simple and vocabulary small
    text = text.lower()
    allowed_chars = "abcdefghijklmnopqrstuvwxyz \n"
    text = "".join([c if c in allowed_chars else " " for c in text])
    text = " ".join(text.split()) # normalize spaces
    return text

def compile_nexa():
    print(f"\n[BUILD] Compiling the high-performance NexaQuant C++ training engine...")
    compile_cmd = ["g++", "-O3", "-mavx2", "-mfma", "main.cpp", "-o", "nexa_bench", "-lpthread"]
    try:
        # Check OS
        if sys.platform.startswith("win"):
            # On windows compile as exe
            compile_cmd[7] = "nexa_bench.exe"
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{GREEN}[SUCCESS] Compilation successful!{RESET}")
            return "nexa_bench.exe" if sys.platform.startswith("win") else "./nexa_bench"
        else:
            print(f"{RED}[ERROR] Compilation failed!{RESET}")
            print(result.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"{RED}[ERROR] Could not run compiler (g++ is required):{RESET} {e}")
        sys.exit(1)

def main():
    print_header()
    
    # 1. Load and prepare raw text
    text = get_or_create_input_text()
    
    # 2. Extract character-level vocabulary
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    
    print(f"[INFO] Vocabulary Size: {vocab_size} unique characters")
    print(f"[INFO] Vocabulary: {vocab}")
    
    # 3. Create sliding-window training samples
    context_len = 5
    in_features = context_len * vocab_size
    out_features = vocab_size
    
    print(f"\n[INFO] Creating training dataset...")
    print(f"  - Context Window: {context_len} characters")
    print(f"  - Input Vector Dimension (One-Hot Concatenation): {in_features}")
    print(f"  - Output Target Vector Dimension: {out_features}")
    
    samples = []
    for i in range(len(text) - context_len):
        context = text[i : i + context_len]
        target = text[i + context_len]
        
        # One-hot encode context
        in_vec = []
        for char in context:
            vec = [0.0] * vocab_size
            vec[char_to_idx[char]] = 1.0
            in_vec.extend(vec)
            
        # One-hot encode target
        out_vec = [0.0] * vocab_size
        out_vec[char_to_idx[target]] = 1.0
        
        samples.append((in_vec, out_vec))
        
    dataset_file = "text_dataset.txt"
    with open(dataset_file, "w") as f:
        f.write(f"# NexaQuant custom text training dataset\n")
        for in_vec, out_vec in samples:
            in_str = " ".join(map(str, in_vec))
            out_str = " ".join(map(str, out_vec))
            f.write(f"{in_str} | {out_str}\n")
            
    print(f"{GREEN}[SUCCESS] Prepared {len(samples)} sliding-window training samples in '{dataset_file}'!{RESET}")
    
    # 4. Compile the engine
    binary_path = compile_nexa()
    
    # 5. Launch Training on C++ NexaQuant Engine
    epochs = 400
    learning_rate = 1.0
    hidden_features = 64
    model_output = "text_predictor.bin"
    
    print(f"\n{CYAN}--- STARTING CPU TERNARY TRAINING PIPELINE (Zero-FP32) ---{RESET}")
    print(f"[RUN] Executing: {binary_path} --train-dataset {dataset_file} {in_features} {hidden_features} {out_features} {epochs} {learning_rate} {model_output}")
    
    # Launch training process and stream output to the user
    train_cmd = [binary_path, "--train-dataset", dataset_file, str(in_features), str(hidden_features), str(out_features), str(epochs), str(learning_rate), model_output]
    process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    if process.returncode != 0:
        print(f"\n{RED}[ERROR] C++ training process failed!{RESET}")
        sys.exit(1)
        
    print(f"\n{GREEN}[CONGRATULATIONS] Your model '{model_output}' was trained successfully using NexaQuant v3.0!{RESET}")
    
    # 6. Interactive Character Generation (Inference)
    print(f"\n{CYAN}========================================================================{RESET}")
    print(f" 🔮 {GREEN}NEXAQUANT INTERACTIVE INFERENCE GENERATION DEMO{RESET}")
    print(f"{CYAN}========================================================================{RESET}")
    
    # Seed prompt
    prompt = "to be"
    if len(prompt) < context_len:
        prompt = prompt.rjust(context_len, " ")
    elif len(prompt) > context_len:
        prompt = prompt[-context_len:]
        
    print(f"\n[INFO] Starting text generation with seed prompt: '{YELLOW}{prompt}{RESET}'")
    print(f"{GREEN}[OUTPUT] {prompt}", end="")
    sys.stdout.flush()
    
    generated_text = prompt
    for _ in range(100):
        # Format input for --predict CLI command
        current_context = generated_text[-context_len:]
        in_vec = []
        for char in current_context:
            vec = [0.0] * vocab_size
            if char in char_to_idx:
                vec[char_to_idx[char]] = 1.0
            in_vec.extend(vec)
            
        predict_cmd = [binary_path, "--predict", model_output] + list(map(str, in_vec))
        
        result = subprocess.run(predict_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n{RED}[ERROR] Inference prediction CLI call failed!{RESET}")
            print(result.stderr)
            break
            
        # Parse prediction output from C++ prints
        # Expected format in stdout: Output: [val1, val2, ... valN]
        out_lines = result.stdout.split("\n")
        output_line = ""
        for line in out_lines:
            if line.startswith("Output: ["):
                output_line = line
                break
                
        if not output_line:
            # Fallback if parsing fails
            next_char = " "
        else:
            try:
                # Extract values between [ and ]
                arr_str = output_line[output_line.find("[") + 1 : output_line.find("]")]
                predictions = list(map(float, arr_str.split(",")))
                
                # Apply high-precision softmax sampling or simply argmax
                # Since NexaQuant outputs are raw activations, let's do a simple weighted sample
                # to make text generation creative and avoid simple loops!
                exp_preds = []
                temp = 0.5 # temperature
                for val in predictions:
                    # Clip to prevent overflow
                    val = max(min(val, 20.0), -20.0)
                    exp_preds.append(max(0.0001, val)) # ReLU output is positive
                
                total = sum(exp_preds)
                probs = [p / total for p in exp_preds]
                
                # Select based on probabilities
                next_idx = random.choices(range(vocab_size), weights=probs, k=1)[0]
                next_char = idx_to_char[next_idx]
            except Exception as e:
                next_char = " "
                
        print(next_char, end="")
        sys.stdout.flush()
        generated_text += next_char
        
    print(f"\n\n{GREEN}[INFO] Generation complete!{RESET}")
    print(f"[INFO] To train on another custom text, write your own text into 'input.txt' and rerun this script!")

if __name__ == "__main__":
    main()
