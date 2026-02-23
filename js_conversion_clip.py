import os
import subprocess
import sys
import torch
import torch.nn as nn
import clip
import onnx2tf

# --- CONFIGURATION ---
TEXT_ONNX_PATH = "clip-text-vit-32.onnx"
IMAGE_ONNX_PATH = "clip-image-vit-32.onnx"

TEXT_TF_DIR = "clip-text-vit-32-tf"
IMAGE_TF_DIR = "clip-image-vit-32-tf"

TEXT_TFJS_DIR = "clip-text-vit-32-tfjs"
IMAGE_TFJS_DIR = "clip-image-vit-32-tfjs"

# --- THE FIX: MANUAL LAYER NORM ---


class ManualLayerNorm(nn.Module):
    def __init__(self, weight, bias, eps):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
        return x_norm


def patch_layernorms(module):
    """Recursively hunts down PyTorch LayerNorms and swaps them with our manual math."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, ManualLayerNorm(
                child.weight, child.bias, child.eps))
        else:
            patch_layernorms(child)


def convert_to_tfjs(saved_model_dir, tfjs_output_dir):
    """Utility to call the TFJS converter shell command."""
    print(f"  -> Converting {saved_model_dir} to TFJS...")
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format", "tf_saved_model",
        saved_model_dir,
        tfjs_output_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  !! TFJS Conversion Failed for {tfjs_output_dir}!")
        print(result.stderr)
    else:
        print(f"  -> Success: {tfjs_output_dir}")


def run_conversion():
    # 1. Load the PyTorch Model
    print("Loading PyTorch CLIP model...")
    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval().float()

    # 2. Patch the entire model (Both text and visual paths use LayerNorm)
    print("Applying LayerNorm patches to the entire model architecture...")
    patch_layernorms(model)

    # --- PART A: TEXT MODEL ---
    print("\n[PART A] Exporting Text Model...")
    dummy_text = clip.tokenize(["a photo of a cat"])

    # We monkeypatch the forward call specifically for the export
    original_forward = model.forward
    model.forward = model.encode_text

    torch.onnx.export(
        model, dummy_text, TEXT_ONNX_PATH,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )
    model.forward = original_forward  # Restore it

    onnx2tf.convert(input_onnx_file_path=TEXT_ONNX_PATH,
                    output_folder_path=TEXT_TF_DIR, non_verbose=True)
    convert_to_tfjs(TEXT_TF_DIR, TEXT_TFJS_DIR)

    # --- PART B: IMAGE MODEL ---
    print("\n[PART B] Exporting Image Model...")
    dummy_image = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model.visual, dummy_image, IMAGE_ONNX_PATH,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=['input'], output_names=['output']
    )

    onnx2tf.convert(input_onnx_file_path=IMAGE_ONNX_PATH,
                    output_folder_path=IMAGE_TF_DIR, non_verbose=True)
    convert_to_tfjs(IMAGE_TF_DIR, IMAGE_TFJS_DIR)

    print("\nAll conversions complete! Check the -tfjs folders.")


if __name__ == "__main__":
    run_conversion()
