# -*- coding: utf-8 -*-
"""
Flask backend for Deep JSCC-Q inference on custom images.
Compatible with TF 2.21 + Keras 3.12 + Python 3.13.

The saved .keras models contain a Lambda layer whose bytecode was compiled on
a different Python version (Colab/3.10), so we CANNOT deserialize them on
Python 3.13. Instead we:
  1. Rebuild the exact same architecture natively (no Lambda — use a subclassed layer).
  2. Extract weights from the .keras zip (model.weights.h5).
  3. Load those weights into our freshly-built model.
"""

import os, io, json, base64, warnings, zipfile, tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
keras = tf.keras
print(f"TF {tf.__version__} | Keras {keras.__version__}", flush=True)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "DeepJSCCQ_Model")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

IMG_H = cfg["IMG_H"]    # 64
IMG_W = cfg["IMG_W"]    # 64
H_LAT = cfg["H_LAT"]   # 16
W_LAT = cfg["W_LAT"]   # 16
C_OUT = cfg["C_OUT"]    # 24
M_QAM = cfg["M"]        # 16
SNR_T = cfg["SNR_DB"]   # 10
IMG_C = 3
N_SRC = IMG_H * IMG_W * IMG_C

# ─── Custom Keras layers ──────────────────────────────────────────────────────

def _make_constellation(M):
    m = int(np.sqrt(M))
    pts = np.arange(-(m-1), m, 2, dtype=np.float32)
    I, Q = np.meshgrid(pts, pts)
    c = np.stack([I.flatten(), Q.flatten()], axis=1)
    return tf.constant(c / np.sqrt(np.mean(np.sum(c**2, axis=1))))


class NormLayer(keras.layers.Layer):
    """Replaces: layers.Lambda(lambda t: t * 2.0 - 1.0)"""
    def call(self, x):
        return x * 2.0 - 1.0
    def get_config(self):
        return super().get_config()


class SoftQuantizer(keras.layers.Layer):
    def __init__(self, M=16, sigma_q=10.0, **kw):
        super().__init__(**kw)
        self.M, self.sigma_q = M, sigma_q
        self.constellation = _make_constellation(M)

    def call(self, z):
        shape = tf.shape(z)
        zf = tf.reshape(z, [-1, 2])
        d  = tf.reduce_sum(tf.square(tf.expand_dims(zf, 1) - self.constellation), axis=2)
        w  = tf.nn.softmax(-self.sigma_q * d, axis=1)
        return tf.reshape(tf.reduce_sum(tf.expand_dims(w, 2) * self.constellation, axis=1), shape)

    def get_config(self):
        return {**super().get_config(), "M": self.M, "sigma_q": self.sigma_q}


class HardQuantizer(keras.layers.Layer):
    def __init__(self, M=16, **kw):
        super().__init__(**kw)
        self.M = M
        self.constellation = _make_constellation(M)

    def call(self, z):
        shape = tf.shape(z)
        zf  = tf.reshape(z, [-1, 2])
        d   = tf.reduce_sum(tf.square(tf.expand_dims(zf, 1) - self.constellation), axis=2)
        idx = tf.argmin(d, axis=1)
        zh  = tf.gather(self.constellation, idx)
        return tf.reshape(tf.stop_gradient(zh - zf) + zf, shape)

    def get_config(self):
        return {**super().get_config(), "M": self.M}


class AWGN_Q(keras.layers.Layer):
    def __init__(self, snr_db=10, **kw):
        super().__init__(**kw)
        self.snr_db = float(snr_db)

    def call(self, z):
        std = tf.sqrt(1.0 / (2.0 * 10.0 ** (self.snr_db / 10.0)))
        return z + tf.random.normal(tf.shape(z), stddev=std)

    def get_config(self):
        return {**super().get_config(), "snr_db": self.snr_db}


# ─── Architecture rebuild ─────────────────────────────────────────────────────

def build_encoder(C_out):
    """Exact replica of training encoder — NormLayer replaces Lambda."""
    layers = keras.layers
    inp = layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x   = NormLayer()(inp)                                            # [0,1]→[-1,1]
    x   = layers.Conv2D(16, 5, strides=2, padding="same")(x)         # 64→32
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    x   = layers.Conv2D(32, 5, strides=2, padding="same")(x)         # 32→16
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    x   = layers.Conv2D(32, 5, strides=1, padding="same")(x)
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    x   = layers.Conv2D(32, 5, strides=1, padding="same")(x)
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    out = layers.Conv2D(C_out, 5, strides=1, padding="same")(x)
    out = layers.PReLU(shared_axes=[1, 2])(out)
    return keras.Model(inp, out, name="encoder")


def build_decoder(C_out):
    """Exact replica of training decoder."""
    layers = keras.layers
    inp = layers.Input(shape=(H_LAT, W_LAT, C_out))
    x   = layers.Conv2DTranspose(32, 5, strides=1, padding="same")(inp)
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    x   = layers.Conv2DTranspose(32, 5, strides=1, padding="same")(x)
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    x   = layers.Conv2DTranspose(32, 5, strides=1, padding="same")(x)
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    x   = layers.Conv2DTranspose(16, 5, strides=2, padding="same")(x)  # 16→32
    x   = layers.PReLU(shared_axes=[1, 2])(x)
    out = layers.Conv2DTranspose(3,  5, strides=2, padding="same",
                                 activation="sigmoid")(x)              # 32→64
    return keras.Model(inp, out, name="decoder")


# ─── Load weights from .keras zip ────────────────────────────────────────────

def load_weights_from_keras_zip(model, keras_path):
    """
    Extract model.weights.h5 from the .keras zip and load into model layer by layer.
    """
    import h5py
    with zipfile.ZipFile(keras_path) as zf:
        if "model.weights.h5" not in zf.namelist():
            raise ValueError(f"model.weights.h5 not found in {keras_path}")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(zf.read("model.weights.h5"))
            tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "r") as f:
            if "layers" not in f: return model
            
            # Map of Keras 3 saved layer names -> their weights
            # The layers have names like 'conv2d', 'conv2d_1' etc.
            # We will just extract all weight arrays in order and apply them to trainable layers.
            all_weights = []
            
            # Get sorted layer names (usually conv2d, conv2d_1, etc.)
            layer_keys = []
            for k in f["layers"].keys():
                if k == "input_layer" or k == "lambda" or k.startswith("p_re_lu"):
                    continue # these don't have weights, or we don't need them
                layer_keys.append(k)
                
            # Naive sort might be tricky since conv2d_10 comes after conv2d_2. Let's rely on model's trainable weights.
            # Actually, Keras weights are just lists of arrays. We can try load_weights with skip_mismatch
            # But earlier it said "Model expected 10 layers, found 0 saved layers." because of h5 format mismatch.
            pass
            
        # Instead of generic load_weights, we can extract the flat list of numpy arrays from the h5 file
        flat_weights = []
        with h5py.File(tmp_path, "r") as f:
            # We use visititems to find all datasets
            def collect_datasets(name, node):
                if isinstance(node, h5py.Dataset):
                    flat_weights.append((name, node[...]))
            f.visititems(collect_datasets)
            
        # Group them by layer using the layer name prefix
        # We know our rebuilt model has the exact same number of Conv2D/Conv2DTranspose layers.
        conv_layers = [l for l in model.layers if isinstance(l, (keras.layers.Conv2D, keras.layers.Conv2DTranspose))]
        
        # Filter arrays that look like kernels or biases (ignore metadata datasets if any)
        weight_arrays = [w[1] for w in flat_weights if len(w[1].shape) > 0]
        
        # Each Conv2D has 2 weight arrays (kernel, bias)
        if len(weight_arrays) == len(conv_layers) * 2:
            idx = 0
            for l in conv_layers:
                l.set_weights([weight_arrays[idx], weight_arrays[idx+1]])
                idx += 2
            print(f"  Loaded {len(weight_arrays)} weight arrays manually into {len(conv_layers)} layers.")
        else:
            print(f"  Warning: Expected {len(conv_layers)*2} weight arrays, found {len(weight_arrays)}")
            
    except Exception as e:
        import traceback
        print(f"Warning: load_weights failed with {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return model


# ─── Build and load ───────────────────────────────────────────────────────────
ENC_FILE = os.path.join(MODEL_DIR, f"encoder_qam{M_QAM}_cout{C_OUT}_snr{SNR_T}.keras")
DEC_FILE = os.path.join(MODEL_DIR, f"decoder_qam{M_QAM}_cout{C_OUT}_snr{SNR_T}.keras")

print("Building encoder architecture…", flush=True)
encoder = build_encoder(C_OUT)
print("Loading encoder weights from .keras zip…", flush=True)
load_weights_from_keras_zip(encoder, ENC_FILE)
print(f"Encoder ready: {encoder.input_shape} -> {encoder.output_shape}", flush=True)

print("Building decoder architecture…", flush=True)
decoder = build_decoder(C_OUT)
print("Loading decoder weights from .keras zip…", flush=True)
load_weights_from_keras_zip(decoder, DEC_FILE)
print(f"Decoder ready: {decoder.input_shape} -> {decoder.output_shape}", flush=True)

_hq = HardQuantizer(M=M_QAM)

# ─── Quick sanity test ────────────────────────────────────────────────────────
_dummy = np.zeros((1, IMG_H, IMG_W, IMG_C), dtype=np.float32)
_z     = encoder(_dummy, training=False).numpy()
_rec   = decoder(_z, training=False).numpy()
print(f"Sanity check: latent={_z.shape}  rec={_rec.shape}", flush=True)

# ─── Flask ───────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def preprocess(pil_img):
    return np.array(
        pil_img.convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR),
        dtype=np.float32
    ) / 255.0

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * np.log10(1.0 / mse) if mse > 1e-10 else 80.0

def ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))

def arr_to_b64(arr_f32, size=None):
    u8  = (np.clip(arr_f32, 0, 1) * 255).astype(np.uint8)
    pil = Image.fromarray(u8)
    if size:
        pil = pil.resize(size, Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def jscc_q_infer(img_f32, snr_db):
    batch  = img_f32[np.newaxis]
    z      = encoder(batch, training=False).numpy()
    z_hard = _hq(z).numpy()
    sigma  = float(np.sqrt(1.0 / (2.0 * 10.0 ** (snr_db / 10.0))))
    y      = z_hard + np.random.normal(0, sigma, z_hard.shape).astype(np.float32)
    return np.clip(decoder(y, training=False).numpy()[0], 0.0, 1.0)

def jpeg_baseline(img_u8, kn, snr_db):
    from scipy.special import erfc
    ber     = 0.5 * erfc(np.sqrt(10.0 ** (snr_db / 10.0)))
    n_bits  = max(8, int(kn * N_SRC))
    n_bytes = n_bits // 8

    lo, hi, best = 1, 95, None
    while lo <= hi:
        mid = (lo + hi) // 2
        buf = io.BytesIO()
        Image.fromarray(img_u8).save(buf, format="JPEG", quality=mid)
        d = buf.getvalue()
        if len(d) <= n_bytes:
            best = d; lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)

    bits = np.unpackbits(np.frombuffer(best, dtype=np.uint8))[:n_bits]
    errs = np.random.binomial(1, ber, len(bits)).astype(np.uint8)
    recv = np.packbits(np.bitwise_xor(bits, errs)).tobytes()
    try:
        pil = Image.open(io.BytesIO(recv)).convert("RGB")
        pil.load()
        return np.array(pil.resize((IMG_W, IMG_H), Image.BILINEAR), dtype=np.float32) / 255.0
    except Exception:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/info")
def api_info():
    kn = C_OUT * H_LAT * W_LAT / N_SRC
    return jsonify({
        "model":           f"Deep JSCC-Q ({M_QAM}-QAM)",
        "c_out":           C_OUT,
        "m_qam":           M_QAM,
        "trained_snr_db":  SNR_T,
        "img_size":        f"{IMG_H}×{IMG_W}",
        "bandwidth_ratio": round(kn, 4),
        "latent_shape":    f"{H_LAT}×{W_LAT}×{C_OUT}",
    })

@app.route("/api/transmit", methods=["POST"])
def api_transmit():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    snr_db    = float(request.form.get("snr_db", 10.0))
    show_jpeg = request.form.get("show_jpeg", "true").lower() == "true"

    try:
        pil_img = Image.open(request.files["file"].stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    orig_w, orig_h = pil_img.size
    img_f32 = preprocess(pil_img)
    img_u8  = (img_f32 * 255).astype(np.uint8)

    rec_jscc = jscc_q_infer(img_f32, snr_db)
    p_jscc   = psnr(img_f32, rec_jscc)
    s_jscc   = ssim(img_f32, rec_jscc)

    disp_size = (min(orig_w, 512), min(orig_h, 512))

    buf = io.BytesIO()
    pil_img.convert("RGB").resize(disp_size, Image.LANCZOS).save(buf, format="PNG")
    orig_b64 = base64.b64encode(buf.getvalue()).decode()

    result = {
        "original_b64": orig_b64,
        "jscc_b64":     arr_to_b64(rec_jscc, disp_size),
        "psnr_jscc":    round(p_jscc, 2),
        "ssim_jscc":    round(s_jscc, 4),
        "snr_db":       snr_db,
    }

    if show_jpeg:
        kn       = C_OUT * H_LAT * W_LAT / N_SRC
        rec_jpeg = jpeg_baseline(img_u8, kn, snr_db)
        result["jpeg_b64"]  = arr_to_b64(rec_jpeg, disp_size)
        result["psnr_jpeg"] = round(psnr(img_f32, rec_jpeg), 2)
        result["ssim_jpeg"] = round(ssim(img_f32, rec_jpeg), 4)

    return jsonify(result)

@app.route("/api/snr_sweep", methods=["POST"])
def api_snr_sweep():
    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    show_jpeg = request.form.get("show_jpeg", "true").lower() == "true"
    snr_min   = float(request.form.get("snr_min", 0))
    snr_max   = float(request.form.get("snr_max", 20))
    snr_step  = float(request.form.get("snr_step", 2))

    try:
        pil_img = Image.open(request.files["file"].stream)
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    img_f32 = preprocess(pil_img)
    img_u8  = (img_f32 * 255).astype(np.uint8)
    kn      = C_OUT * H_LAT * W_LAT / N_SRC

    snrs   = [float(s) for s in np.arange(snr_min, snr_max + snr_step, snr_step)]
    p_jscc, s_jscc, p_jpeg, s_jpeg = [], [], [], []

    for snr in snrs:
        r = jscc_q_infer(img_f32, snr)
        p_jscc.append(round(psnr(img_f32, r), 2))
        s_jscc.append(round(ssim(img_f32, r), 4))
        if show_jpeg:
            rj = jpeg_baseline(img_u8, kn, snr)
            p_jpeg.append(round(psnr(img_f32, rj), 2))
            s_jpeg.append(round(ssim(img_f32, rj), 4))

    return jsonify({
        "snr_values": snrs,
        "psnr_jscc":  p_jscc,
        "ssim_jscc":  s_jscc,
        "psnr_jpeg":  p_jpeg,
        "ssim_jpeg":  s_jpeg,
    })


if __name__ == "__main__":
    print("\nDeep JSCC-Q Demo - http://localhost:5000", flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
