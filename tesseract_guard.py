"""
Tesseract Guard: High-Dimensional Logit Steering & Runtime Vector Clamping Engine
Author: Saul Delgado (Tesseract Lens AI)
License: GNU AGPLv3
"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F

class TesseractGuard:
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activations = {}
        self.hooks = []

    def _register_hooks(self, target_layers):
        """Helper to attach forward hooks across transformer hidden layers."""
        self._clear_hooks()
        for layer_idx in target_layers:
            # Flexible handle for Transformer models (Gemma, Qwen, Llama)
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                layer = self.model.model.layers[layer_idx]
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                layer = self.model.transformer.h[layer_idx]
            else:
                raise AttributeError("Unsupported model layer architecture.")

            def get_activation(idx):
                def hook(module, input, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    self.activations[idx] = tensor.detach()
                return hook

            self.hooks.append(layer.register_forward_hook(get_activation(layer_idx)))

    def _clear_hooks(self):
        """Remove active forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    # ---------------------------------------------------------------
    # MODULE 1: Logit Lens Interceptor
    # ---------------------------------------------------------------
    def run_logit_lens(self, prompt, target_layers):
        """Interprets intermediate layer representations directly through the LM head."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._register_hooks(target_layers)

        with torch.no_grad():
            _ = self.model(**inputs)

        results = {}
        norm_layer = getattr(self.model.model, "norm", None) or getattr(self.model, "lm_head", None)
        lm_head = self.model.lm_head

        for layer_idx in target_layers:
            act = self.activations[layer_idx][:, -1, :]
            if norm_layer is not None and hasattr(self.model.model, "norm"):
                act = self.model.model.norm(act)
            logits = lm_head(act)
            top_token_id = torch.argmax(logits, dim=-1).item()
            token_str = self.tokenizer.decode([top_token_id])
            results[layer_idx] = token_str

        self._clear_hooks()
        return results

    # ---------------------------------------------------------------
    # MODULE 2: Shannon Entropy & Variance Auditor
    # ---------------------------------------------------------------
    def audit_layer_entropy(self, prompt, target_layers):
        """Calculates Shannon entropy and variance across intermediate logit projections."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._register_hooks(target_layers)

        with torch.no_grad():
            _ = self.model(**inputs)

        metrics = {}
        lm_head = self.model.lm_head

        for layer_idx in target_layers:
            act = self.activations[layer_idx][:, -1, :].float()
            logits = lm_head(act)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()
            variance = torch.var(act, dim=-1).item()
            metrics[layer_idx] = {"entropy": entropy, "variance": variance}

        self._clear_hooks()
        return metrics

    # ---------------------------------------------------------------
    # MODULE 3: 4D Hypercube Spatial Node Mapper
    # ---------------------------------------------------------------
    def map_4d_spatial_nodes(self, prompt, target_layers):
        """Projects hidden state activations into 4D hypercube coordinates (x, y, z, w)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._register_hooks(target_layers)

        with torch.no_grad():
            _ = self.model(**inputs)

        layer_vectors = []
        valid_layers = []

        for layer_idx in target_layers:
            vec = self.activations[layer_idx][0, -1, :].float().cpu().numpy()
            layer_vectors.append(vec)
            valid_layers.append(layer_idx)

        matrix = np.array(layer_vectors)
        pca = PCA(n_components=min(4, matrix.shape[0]))
        coords_4d = pca.fit_transform(matrix)

        spatial_map = {}
        for idx, layer_idx in enumerate(valid_layers):
            c = coords_4d[idx]
            # Pad to 4D if layer count is fewer than 4
            padded = np.pad(c, (0, 4 - len(c)), mode='constant') if len(c) < 4 else c[:4]
            spatial_map[layer_idx] = {
                "x": float(padded[0]),
                "y": float(padded[1]),
                "z": float(padded[2]),
                "w": float(padded[3])
            }

        self._clear_hooks()
        return spatial_map

    # ---------------------------------------------------------------
    # MODULE 4: Directed Vector Steering Engine
    # ---------------------------------------------------------------
    def compute_concept_vector(self, positive_prompt, negative_prompt, steer_layer):
        """Extracts a semantic direction vector (Pos - Neg) at a specified layer."""
        pos_in = self.tokenizer(positive_prompt, return_tensors="pt").to(self.device)
        neg_in = self.tokenizer(negative_prompt, return_tensors="pt").to(self.device)

        self._register_hooks([steer_layer])
        with torch.no_grad():
            _ = self.model(**pos_in)
            pos_act = self.activations[steer_layer][:, -1, :].clone().float()

            _ = self.model(**neg_in)
            neg_act = self.activations[steer_layer][:, -1, :].clone().float()
        self._clear_hooks()

        concept_vector = pos_act - neg_act
        return concept_vector

    def generate_steered(self, prompt, concept_vector, steer_layer, alpha=1.0, max_new_tokens=40):
        """Applies directed spatial vector steering during sequence generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                steered_tensor = output[0].clone()
                steered_tensor[:, -1, :] = steered_tensor[:, -1, :] + (alpha * concept_vector.to(steered_tensor.dtype))
                return (steered_tensor,) + output[1:]
            else:
                steered_tensor = output.clone()
                steered_tensor[:, -1, :] = steered_tensor[:, -1, :] + (alpha * concept_vector.to(steered_tensor.dtype))
                return steered_tensor

        if hasattr(self.model, "model"):
            target_module = self.model.model.layers[steer_layer]
        else:
            target_module = self.model.transformer.h[steer_layer]

        hook_handle = target_module.register_forward_hook(steering_hook)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        hook_handle.remove()
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ---------------------------------------------------------------
    # MODULE 5 & 6: Vector Clamping & Entropy Stabilization Engine
    # ---------------------------------------------------------------
    def generate_with_tesseract_guard(
        self, 
        prompt, 
        concept_vector=None, 
        steer_layer=18, 
        alpha=0.0, 
        clamp_layer=19, 
        divergence_threshold=500.0, 
        max_new_tokens=40
    ):
        """
        Runs generation with continuous layer auditing. If vector steering or an adversarial shift 
        causes layer activation variance/entropy to exceed `divergence_threshold`, Module 6 triggers
        an automated clamp to project activations back to baseline bounds.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 1. Capture baseline activation magnitude at clamp layer
        self._register_hooks([clamp_layer])
        with torch.no_grad():
            _ = self.model(**inputs)
            baseline_act = self.activations[clamp_layer][:, -1, :].clone().float()
            baseline_norm = torch.norm(baseline_act, p=2).item()
        self._clear_hooks()

        audit_log = []

        # 2. Define the combined Steering + Clamping hook
        def guardrail_hook(module, input, output):
            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output
            modified = tensor.clone()

            # Execute Module 4 Steering if applicable at this layer
            if module == steer_target and concept_vector is not None and alpha != 0.0:
                modified[:, -1, :] = modified[:, -1, :] + (alpha * concept_vector.to(modified.dtype))

            # Execute Module 6 Clamping if applicable at this layer
            if module == clamp_target:
                current_act = modified[:, -1, :].float()
                current_norm = torch.norm(current_act, p=2).item()
                norm_delta = abs(current_norm - baseline_norm)

                if norm_delta > divergence_threshold:
                    # Clamp vector scale back to baseline magnitude
                    scaling_factor = baseline_norm / (current_norm + 1e-8)
                    clamped_act = current_act * scaling_factor
                    modified[:, -1, :] = clamped_act.to(modified.dtype)
                    audit_log.append({
                        "status": "CLAMPED",
                        "raw_norm_delta": norm_delta,
                        "clamped_norm": torch.norm(clamped_act, p=2).item()
                    })
                else:
                    audit_log.append({
                        "status": "PASS",
                        "raw_norm_delta": norm_delta,
                        "clamped_norm": current_norm
                    })

            return (modified,) + output[1:] if is_tuple else modified

        if hasattr(self.model, "model"):
            steer_target = self.model.model.layers[steer_layer]
            clamp_target = self.model.model.layers[clamp_layer]
        else:
            steer_target = self.model.transformer.h[steer_layer]
            clamp_target = self.model.transformer.h[clamp_layer]

        h1 = steer_target.register_forward_hook(guardrail_hook)
        h2 = clamp_target.register_forward_hook(guardrail_hook) if clamp_layer != steer_layer else None

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        h1.remove()
        if h2:
            h2.remove()

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {
            "text": generated_text,
            "baseline_norm": baseline_norm,
            "audit_log": audit_log
        }

