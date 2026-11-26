import json
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (224, 224)

LABEL_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}


def resolve_model_path() -> str:
    for name in ("bestmodel_lite.keras", "best_brain_mri_model.keras", "bestmodel.keras"):
        candidate = os.path.join(BASE_DIR, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("No .keras model file found in the repository root.")


def load_class_names(num_outputs: int) -> list[str]:
    classes_path = os.path.join(BASE_DIR, "class_names.json")
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            names = json.load(f)
            if len(names) == num_outputs:
                return names
    if num_outputs == 4:
        return ["glioma", "meningioma", "notumor", "pituitary"]
    return [f"Class {i}" for i in range(num_outputs)]


# ==============================================================================
# 1. LOAD MODEL & BUILD GRAPH-SAFE GRAD-CAM MODEL ONCE
# ==============================================================================
@st.cache_resource
def load_cnn_model():
    model_path = resolve_model_path()
    model = tf.keras.models.load_model(model_path, compile=False)

    last_conv = next(
        layer.name
        for layer in reversed(model.layers)
        if isinstance(layer, tf.keras.layers.Conv2D)
    )

    # Use a Functional graph so skip/merge layers (Add, Multiply, etc.) execute properly
    if isinstance(model, tf.keras.Sequential):
        inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = inp
        conv_out = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv:
                conv_out = x
        grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out, x, x])
    else:
        layer_names = [layer.name for layer in model.layers]
        score_tensor = (
            model.get_layer("logits").output
            if "logits" in layer_names
            else model.outputs[0]
        )
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv).output,
                score_tensor,
                model.outputs[0],
            ],
        )

    # EfficientNet models include an internal Rescaling layer and expect [0, 255] inputs
    has_internal_rescaling = any(
        isinstance(layer, tf.keras.layers.Rescaling) for layer in model.layers
    )
    return model, grad_model, last_conv, has_internal_rescaling


model, grad_model, LAST_CONV_LAYER, HAS_INTERNAL_RESCALING = load_cnn_model()


# ==============================================================================
# 2. PREPROCESSING & GRAD-CAM
# ==============================================================================
def auto_crop_brain(pil_img: Image.Image) -> Image.Image:
    normalized = ImageOps.autocontrast(pil_img.convert("RGB"), cutoff=1)
    gray = np.asarray(normalized.convert("L"))
    thresh = max(20, int(np.percentile(gray, 25)) + 10)
    mask = gray > thresh
    if not np.any(mask):
        return normalized

    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1
    h, w = gray.shape
    pad_y, pad_x = int((y_max - y_min) * 0.02), int((x_max - x_min) * 0.02)
    return normalized.crop((
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(w, x_max + pad_x),
        min(h, y_max + pad_y),
    ))


def preprocess_image(pil_img: Image.Image, crop_borders: bool) -> tuple[Image.Image, np.ndarray]:
    img = auto_crop_brain(pil_img) if crop_borders else pil_img.convert("RGB")
    resized = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)
    if not HAS_INTERNAL_RESCALING:
        arr = arr / 255.0
    return resized, np.expand_dims(arr, axis=0)


def apply_jet_colormap(heatmap_01: np.ndarray) -> np.ndarray:
    r = np.clip(1.5 - np.abs(4.0 * heatmap_01 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * heatmap_01 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * heatmap_01 - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def predict_with_gradcam(resized_pil: Image.Image, input_tensor: np.ndarray, overlay_strength: float):
    img_tf = tf.convert_to_tensor(input_tensor)

    with tf.GradientTape() as tape:
        tape.watch(img_tf)
        conv_outputs, scores, predictions = grad_model(img_tf, training=False)
        num_outputs = int(predictions.shape[-1])
        if num_outputs == 1:
            target_score = scores[:, 0]
        else:
            top_idx = int(tf.argmax(predictions[0]))
            target_score = scores[:, top_idx]

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        heatmap_np = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.maximum(tf.squeeze(conv_outputs[0] @ pooled_grads[..., tf.newaxis]), 0)
        max_val = float(tf.reduce_max(heatmap))
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

    heatmap_uint8 = (heatmap_np * 255.0).astype(np.uint8)
    heatmap_resized = (
        np.asarray(Image.fromarray(heatmap_uint8).resize(IMG_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
        / 255.0
    )

    jet_rgb = apply_jet_colormap(heatmap_resized).astype(np.float32)
    orig_rgb = np.asarray(resized_pil, dtype=np.float32)
    alpha = (heatmap_resized[..., np.newaxis] ** 0.85) * 0.65 * overlay_strength
    blended = np.clip(orig_rgb * (1.0 - alpha) + jet_rgb * alpha, 0, 255).astype(np.uint8)

    return predictions[0].numpy(), Image.fromarray(blended)


# ==============================================================================
# 3. FRONTEND UI
# ==============================================================================
st.title("Brain MRI Tumor Classifier")
st.caption("CNN MRI Classification (224×224 RGB) with Grad-CAM Attention")

left_col, right_col = st.columns([1, 1.3], gap="large")

with left_col:
    uploaded = st.file_uploader("Upload an MRI scan (JPG, PNG)", type=["jpg", "jpeg", "png"])
    crop_borders = st.checkbox("Auto-crop dark borders & normalize contrast", value=True)
    overlay_strength = st.slider("Grad-CAM Overlay Opacity", 0.0, 1.0, 1.0, 0.05)

    if uploaded:
        raw_img = Image.open(uploaded)
        st.image(raw_img, caption=f"{uploaded.name}", use_container_width=True)

with right_col:
    if not uploaded:
        st.info("Upload a brain MRI image on the left to view predictions and Grad-CAM heatmaps.")
    else:
        with st.spinner("Running CNN inference & Grad-CAM..."):
            resized_pil, tensor = preprocess_image(raw_img, crop_borders)
            preds, overlay_img = predict_with_gradcam(resized_pil, tensor, overlay_strength)

        if len(preds) == 1:
            raw_score = float(preds[0])
            tumor_prob = raw_score
            no_tumor_prob = 1.0 - raw_score
            is_tumor = tumor_prob >= 0.5
            confidence = (tumor_prob if is_tumor else no_tumor_prob) * 100

            if is_tumor:
                st.error(f"**Tumor Detected** — {confidence:.2f}% confidence")
            else:
                st.success(f"**No Tumor** — {confidence:.2f}% confidence")

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(resized_pil, caption="Model Input (224×224)", use_container_width=True)
            with img_c2:
                st.image(overlay_img, caption=f"Grad-CAM ({LAST_CONV_LAYER})", use_container_width=True)

            st.write(f"**Tumor:** {tumor_prob * 100:.2f}%")
            st.progress(min(max(tumor_prob, 0.0), 1.0))

            st.write(f"**No Tumor:** {no_tumor_prob * 100:.2f}%")
            st.progress(min(max(no_tumor_prob, 0.0), 1.0))

            st.caption(f"Layer: `{LAST_CONV_LAYER}` · Raw Sigmoid: `{raw_score:.6f}` · Educational prototype only.")
        else:
            class_names = load_class_names(len(preds))
            top_idx = int(np.argmax(preds))
            top_key = class_names[top_idx]
            top_label = LABEL_DISPLAY.get(top_key.lower(), top_key.title())
            has_tumor = top_key.lower() != "notumor"
            confidence = float(preds[top_idx]) * 100

            if has_tumor:
                st.error(f"**Tumor Detected · {top_label}** — {confidence:.2f}% confidence")
            else:
                st.success(f"**No Tumor** — {confidence:.2f}% confidence")

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(resized_pil, caption="Model Input (224×224)", use_container_width=True)
            with img_c2:
                st.image(overlay_img, caption=f"Grad-CAM ({LAST_CONV_LAYER})", use_container_width=True)

            ranked = sorted(zip(class_names, preds), key=lambda item: float(item[1]), reverse=True)
            for name, prob in ranked:
                display = LABEL_DISPLAY.get(name.lower(), name.title())
                p = float(prob)
                st.write(f"**{display}:** {p * 100:.2f}%")
                st.progress(min(max(p, 0.0), 1.0))

            st.caption(f"Layer: `{LAST_CONV_LAYER}` · Educational prototype only.")
