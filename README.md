# TesseractLensGuard

**TesseractLensGuard** is an open-source AI safety and steering framework designed to audit, monitor, and clamp high-dimensional vector representations across open-weights transformer models (Gemma, Qwen, Llama, DeepSeek, Mistral)[span_4](start_span)[span_4](end_span)[span_5](start_span)[span_5](end_span).

By intercepting activations across intermediate transformer layers, Tesseract Guard provides real-time visibility into internal logic states, enables directed spatial vector steering, and dynamically clamps activation drifts back to baseline entropy bounds during generation[span_6](start_span)[span_6](end_span)[span_7](start_span)[span_7](end_span).

---

## 🛡️ Module 6: Baseline Vector Clamping & Anti-Hijacking Guardrail

Module 6 introduces dynamic layer-norm monitoring during sequence generation[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span). When unexpected activation shifts or adversarial vector steering push internal representations beyond allowable variance thresholds, the **Baseline Clamping Engine** automatically intercepts the vector delta and clamps activation magnitudes back to baseline bounds[span_10](start_span)[span_10](end_span)[span_11](start_span)[span_11](end_span).

* **Prevents Model Hijacking:** Clamps extreme representation drifts caused by adversarial prompts, jailbreaks, or out-of-distribution vector manipulation[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span).
* **Preserves Generation Stability:** Prevents entropy explosions and infinite repetition loops without requiring full model retraining or re-prompting[span_14](start_span)[span_14](end_span)[span_15](start_span)[span_15](end_span).
* **Enforces Output Safety:** Provides real-time, deterministic bounds to keep model output aligned with safety baseline boundaries[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span).

---

## 🛠️ Installation

Clone the repository and install dependencies directly:

```bash
git clone [https://github.com/TesseractGuard/TesseractLensGuard.git](https://github.com/TesseractGuard/TesseractLensGuard.git)
cd TesseractLensGuard
pip install torch numpy scikit-learn transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tesseract_guard import TesseractGuard

# 1. Initialize Tesseract Guard with target open-weights model
model_id = "unsloth/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

guard = TesseractGuard(model, tokenizer)

# ---------------------------------------------------------------
# 2. Run Layer Audits (Modules 1–3)
# ---------------------------------------------------------------

# Module 1: Logit Lens Interceptor
logit_results = guard.run_logit_lens(
    prompt="The speed of light in a vacuum is", 
    target_layers=[12, 16, 20, 24]
)
print("Logit Lens Output:", logit_results)

# Module 2: Shannon Entropy & Variance Audit
entropy_metrics = guard.audit_layer_entropy(
    prompt="What is quantum entanglement?", 
    target_layers=[10, 15, 20]
)
print("Entropy Audit:", entropy_metrics)

# Module 3: 4D Hypercube Spatial Node Mapping
spatial_map = guard.map_4d_spatial_nodes(
    prompt="Explain neural network steering.", 
    target_layers=[4, 8, 12, 16]
)
print("4D Coordinates:", spatial_map)

# ---------------------------------------------------------------
# 3. Vector Steering & Anti-Hijacking Vector Clamping (Modules 4–6)
# ---------------------------------------------------------------

# Extract semantic direction vector (Pos - Neg) at Layer 18
concept_vector = guard.compute_concept_vector(
    positive_prompt="encrypted cryptographic security key",
    negative_prompt="simple numeric counter digit",
    steer_layer=18
)

# Execute generation with Module 6 Baseline Clamping active
guarded_response = guard.generate_with_tesseract_guard(
    prompt="The secret key to the vault is",
    concept_vector=concept_vector,
    steer_layer=18,
    alpha=8.0,                  # Steering multiplier
    clamp_layer=19,              # Layer to monitor and clamp
    divergence_threshold=500.0,  # Max norm divergence before anti-hijack clamp triggers
    max_new_tokens=40
)

print("Guarded Output:", guarded_response["text"])
print("Baseline Layer Norm:", guarded_response["baseline_norm"])
print("Audit Log:", guarded_response["audit_log"])
