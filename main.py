# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
import cv2
import matplotlib.cm as cm
import tensorflow as tf
from keras.models import load_model
from keras.utils import img_to_array
import io

# ================= Load Model =================
MODEL_PATH = "real_vs_fake_model.h5"
model = load_model(MODEL_PATH)
LAST_CONV_LAYER = "conv5_block16_2_conv"  # تأكد إنه مظبوط

# ================= FastAPI =================
app = FastAPI(title="Real vs Fake Face Detection API")

# ================= Prediction Function =================
def predict_with_confidence(model, img_array):
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = (1 - pred) * 100 if label == "Fake" else pred * 100
    return label, pred, confidence

# ================= Grad-CAM Function =================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if preds[0][0] < 0.5:
        grads = -grads

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# ================= Apply Grad-CAM & Boxes =================
def apply_gradcam_refined(img_array, heatmap, alpha=0.5, intensity_threshold=0.7, label_text="Real", confidence=100):
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    heatmap_res = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))

    mask = np.uint8(heatmap_res > intensity_threshold * np.max(heatmap_res)) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxed_img = img_rgb.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 15 and h > 15:
            cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # ضع النص أعلى الصورة
    text = f"{label_text} | {confidence:.2f}%"
    cv2.putText(boxed_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    heatmap_uint8 = np.uint8(255 * heatmap_res)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = np.uint8(jet_colors[heatmap_uint8] * 255)
    superimposed_img = cv2.addWeighted(jet_heatmap, alpha, img_rgb, 1-alpha, 0)

    return superimposed_img, boxed_img

# ================= Helper: Convert to bytes =================
def image_to_bytes(img):
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return io.BytesIO(buffer.tobytes())

# ================= API Endpoint =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (256, 256))
    img_array = np.expand_dims(img_to_array(tf.keras.preprocessing.image.array_to_img(img_resized)), axis=0) / 255.

    # Prediction
    label, pred, confidence = predict_with_confidence(model, img_array)
    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
    superimposed_img, boxed_img = apply_gradcam_refined(img, heatmap, label_text=label, confidence=confidence)

    # Return image مع النص
    buffered = image_to_bytes(boxed_img)
    return StreamingResponse(buffered, media_type="image/jpeg", headers={
        "X-Prediction": label,
        "X-Confidence": f"{confidence:.2f}%"
    })

# ================= Run App =================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
