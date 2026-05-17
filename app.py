import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# -----------------------
# MODEL DEFINITION
# -----------------------
class SEBlock(nn.Module):
    def __init__(self, ch, ratio=4):
        super().__init__()
        mid = max(ch // ratio, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
        self.se = SEBlock(ch)
    def forward(self, x):
        return x + self.se(self.net(x))

class EMNISTNet(nn.Module):
    def __init__(self, num_classes=26, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResBlock(64), ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            ResBlock(128), ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.stage3 = nn.Sequential(
            ResBlock(256, 0.15), ResBlock(256, 0.15),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, 26),
        )
    def forward(self, x):
        return self.head(self.stage3(self.stage2(self.stage1(self.stem(x)))))

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    model = EMNISTNet()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------
# PREPROCESSING FUNCTIONS
# -----------------------
MEAN = 0.1722
STD  = 0.3309

def preprocess_drawn_image(pil_img):
    img = pil_img.convert("L")

    # 🔥 ALWAYS invert (no checkbox needed)
    img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.BILINEAR)

    arr = np.array(img).astype(np.float32)

    arr = arr / 255.0
    arr = (arr - 0.1722) / 0.3309

    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    return tensor

def preprocess_emnist_csv_flat_image(pil_img):
    """If user provides raw EMNIST CSV-derived image (flattened), apply the
       same rot/flip fix you used during training before normalizing."""
    img = pil_img.convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    # apply CSV fix identical to training
    arr = np.rot90(arr, k=3)
    arr = np.fliplr(arr)
    arr = arr / 255.0
    arr = (arr - MEAN) / STD
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor

# -----------------------
# UI
# -----------------------
st.title("🔤 EMNIST Letter Recognition — DRAW & UPLOAD")
st.write("Choose input mode and tune options if predictions look off.")

input_mode = st.radio("Input Mode", ["Draw (canvas)", "Upload image (user photo)", "Raw EMNIST CSV image (use CSV fix)"])
show_topk = st.slider("Top-k predictions", value=3, min_value=1, max_value=6)

# Drawing canvas (always available)
st.subheader("Draw a letter")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Upload option
uploaded_file = st.file_uploader("Or upload an image (png/jpg)", type=["png","jpg","jpeg"])

# Choose the image source to use for inference
img_for_infer = None
if input_mode == "Draw (canvas)":
    if canvas_result.image_data is not None:
        # canvas_result.image_data is RGBA-like HxWx4 float array 0..255
        arr = canvas_result.image_data.astype(np.uint8)
        # take the alpha-composited result; just convert to PIL using all channels
        pil = Image.fromarray(arr)
        # convert to grayscale later in preprocess
        img_for_infer = ("draw", pil)
elif uploaded_file is not None:
    pil = Image.open(uploaded_file).convert("RGBA")
    # if user selected raw EMNIST CSV fix mode, treat it differently
    if input_mode == "Raw EMNIST CSV image (use CSV fix)":
        img_for_infer = ("csv", pil.convert("L"))
    else:
        img_for_infer = ("upload", pil.convert("L"))

if img_for_infer is not None:
    tag, pil_img = img_for_infer
    st.image(pil_img, caption="Input image (preview)", width=200)
    # Preprocess according to chosen mode
    if tag == "draw" or tag == "upload":
        inp = preprocess_drawn_image(pil_img)
    elif tag == "csv":
        inp = preprocess_emnist_csv_flat_image(pil_img)

    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        topk_idx = probs.argsort()[::-1][:show_topk]
        topk_vals = probs[topk_idx]

    # Present top-k
    st.subheader("Predictions")
    for i, (idx, p) in enumerate(zip(topk_idx, topk_vals)):
        letter = chr(int(idx) + 65)
        st.write(f"{i+1}. {letter} — {p*100:.2f}%")

    st.success(f"Top prediction: {chr(int(topk_idx[0]) + 65)} ({topk_vals[0]*100:.2f}%)")
else:
    st.info("Draw in the canvas above or upload an image to get predictions.")