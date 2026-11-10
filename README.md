# TesseractLensGuard

**TesseractLensGuard** is an open-source AI safety and steering framework designed to audit, monitor, and clamp high-dimensional vector representations across open-weights transformer models (Gemma, Qwen, Llama, DeepSeek, Mistral).

By intercepting activations across intermediate transformer layers, Tesseract Guard provides real-time visibility into internal logic states, enables directed spatial vector steering, and dynamically clamps activation drifts back to baseline entropy bounds during generation.

---

## 🛠️ Installation

Clone the repository and install dependencies directly:

```bash
git clone [https://github.com/TesseractLensAI/TesseractLensGuard.git](https://github.com/TesseractLensAI/TesseractLensGuard.git)
cd TesseractLensGuard
pip install torch numpy scikit-learn transformers
