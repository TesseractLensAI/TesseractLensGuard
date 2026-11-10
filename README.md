# TesseractLensGuard

**TesseractLensGuard** is an open-source AI safety and steering framework designed to audit, monitor, and clamp high-dimensional vector representations across open-weights transformer models (Gemma, Qwen, Llama, DeepSeek, Mistral).

By intercepting activations across intermediate transformer layers, Tesseract Guard provides real-time visibility into internal logic states, enables directed spatial vector steering, and dynamically clamps activation drifts back to baseline entropy bounds during generation.

---

## 🛡️ Module 6: Baseline Vector Clamping & Anti-Hijacking Guardrail

Module 6 introduces dynamic layer-norm monitoring during sequence generation. When unexpected activation shifts or adversarial vector steering push internal representations beyond allowable variance thresholds, the **Baseline Clamping Engine** automatically intercepts the vector delta and clamps activation magnitudes back to baseline bounds.

* **Prevents Model Hijacking:** Clamps extreme representation drifts caused by adversarial prompts, jailbreaks, or out-of-distribution vector manipulation.
* **Preserves Generation Stability:** Prevents entropy explosions and infinite repetition loops without requiring full model retraining or re-prompting.
* **Enforces Output Safety:** Provides real-time, deterministic bounds to keep model output aligned with safety baseline boundaries.

---

## 🛠️ Installation

Clone the repository and install dependencies directly:

```bash
git clone [https://github.com/TesseractLensAI/TesseractLensGuard.git](https://github.com/TesseractLensAI/TesseractLensGuard.git)
cd TesseractLensGuard
pip install torch numpy scikit-learn transformers
