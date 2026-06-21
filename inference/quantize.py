"""
Quantization pipeline: PyTorch training checkpoint -> ONNX -> TFLite
(post-training int8 quantization).

Per the brief's guidance: "Quantize your PyTorch models to TFLite to
optimize for CPU-based inference." Int8 quantization typically gives a
3-4x size reduction and a comparable inference speedup on CPU vs FP32,
which is what makes the <2.0s edge-friendly SLA realistic on commodity
GKE nodes (no GPU required).

This is a reference script — run it offline as part of the training/
release pipeline, not inside a request path. It expects a representative
calibration dataset (a few hundred real field images per crop) for the
int8 calibration step; using random data here would produce a
technically-valid but poorly-calibrated (low F1) model.

Usage:
    python -m inference.quantize \
        --checkpoint checkpoints/cassava_resnet18.pt \
        --calibration-dir data/calibration/cassava \
        --output models/cassava_v3_int8.tflite
"""
import argparse
import pathlib

import numpy as np
import onnx
import tensorflow as tf
import torch


def export_to_onnx(checkpoint_path: str, onnx_path: str, input_size: int) -> None:
    model = torch.load(checkpoint_path, map_location="cpu")
    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None,  # fixed batch size of 1 — matches serving shape
    )
    onnx.checker.check_model(onnx.load(onnx_path))


def _representative_dataset(calibration_dir: str, input_size: int):
    """Yields calibration samples for TFLite's int8 range estimation.
    Must be real, representative field photos — not synthetic noise —
    or the quantized model's F1-score will silently degrade below the
    85% floor required by the brief."""
    from PIL import Image

    paths = sorted(pathlib.Path(calibration_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(
            f"no calibration images found in {calibration_dir} — int8 "
            "quantization requires a representative sample, not synthetic data"
        )
    for path in paths[:200]:  # a few hundred images is typically sufficient
        img = Image.open(path).convert("RGB").resize((input_size, input_size))
        array = (np.asarray(img, dtype=np.float32) / 255.0)[np.newaxis, ...]
        yield [array]


def convert_to_tflite_int8(
    onnx_path: str, calibration_dir: str, output_path: str, input_size: int
) -> None:
    # Bridge step: ONNX -> SavedModel via onnx-tf, then SavedModel -> TFLite.
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = str(pathlib.Path(output_path).with_suffix("")) + "_savedmodel"
    tf_rep.export_graph(saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        calibration_dir, input_size
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(output_path).write_bytes(tflite_model)
    print(f"wrote quantized model to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=224)
    args = parser.parse_args()

    onnx_path = str(pathlib.Path(args.output).with_suffix(".onnx"))
    export_to_onnx(args.checkpoint, onnx_path, args.input_size)
    convert_to_tflite_int8(onnx_path, args.calibration_dir, args.output, args.input_size)

    # Carry the labels sidecar produced by training/train.py through to
    # the served artifact, so inference/model_server/app.py always loads
    # the exact class list this model was trained on rather than relying
    # on a hand-maintained list that can silently drift out of sync.
    labels_src = pathlib.Path(f"{args.checkpoint}.labels.json")
    if labels_src.exists():
        labels_dst = pathlib.Path(f"{args.output}.labels.json")
        labels_dst.write_text(labels_src.read_text())
        print(f"copied labels sidecar to {labels_dst}")
    else:
        print(
            f"warning: no labels sidecar found at {labels_src} — "
            "model_server/app.py will fall back to its hardcoded "
            "CLASS_LABELS for this model version, which may not match."
        )


if __name__ == "__main__":
    main()
