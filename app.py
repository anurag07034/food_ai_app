import streamlit as st
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import json
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food AI", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🍽️ Food AI")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# -----------------------------
# LOAD DATA
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

nutrition_df = pd.read_csv("nutrition_ifct_40.csv")
nutrition_df.columns = nutrition_df.columns.str.lower().str.strip()

# Auto-detect columns safely
food_col = nutrition_df.columns[0]
cal_col = nutrition_df.columns[1]
prot_col = nutrition_df.columns[2]
carb_col = nutrition_df.columns[3]
fat_col = nutrition_df.columns[4]

nutrition_df[food_col] = nutrition_df[food_col].str.lower().str.strip()

# -----------------------------
# MODELS
# -----------------------------
effnet_model = torchvision.models.efficientnet_b0(weights=None)
effnet_model.classifier[1] = torch.nn.Linear(1280, 40)
effnet_model.load_state_dict(torch.load("effnet_food40_final.pth", map_location=device))
effnet_model.to(device)
effnet_model.eval()

mask_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
mask_model.to(device)
mask_model.eval()

# -----------------------------
# TRANSFORMS
# -----------------------------
classification_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

# -----------------------------
# DEPTH
# -----------------------------
def get_depth_map(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    depth = cv2.GaussianBlur(gray, (11, 11), 0)
    return depth.astype("float32") / 255.0

# -----------------------------
# MAIN
# -----------------------------
st.title("🍽️ Food AI - Smart Nutrition Estimator")

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img_rgb = img.copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

    # -----------------------------
    # SEGMENTATION
    # -----------------------------
    img_resized = cv2.resize(img_rgb, (512, 512))

    with torch.no_grad():
        predictions = mask_model([mask_transform(img_resized).to(device)])

    masks = predictions[0]["masks"]
    scores = predictions[0]["scores"]

    valid_masks = []

    for i in range(len(masks)):
        if scores[i] < 0.5:
            continue

        mask = masks[i, 0].cpu().numpy()
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        mask_binary = (mask > 0.5).astype("uint8")

        if mask_binary.sum() > 0.8 * (img.shape[0] * img.shape[1]):
            continue

        kernel = np.ones((5,5), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_binary = (labels == largest).astype("uint8")

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([15, 80, 80])
        upper = np.array([40, 255, 255])
        color_mask = cv2.inRange(hsv, lower, upper)

        mask_binary = mask_binary & (color_mask > 0)
        mask_binary = mask_binary.astype("uint8")

        mask_binary = cv2.medianBlur(mask_binary, 5)
        mask_binary = cv2.GaussianBlur(mask_binary.astype("float32"), (5,5), 0)
        mask_binary = (mask_binary > 0.3).astype("uint8")

        if mask_binary.sum() < 3000:
            continue

        valid_masks.append(mask_binary)

    valid_masks = sorted(valid_masks, key=lambda x: x.sum(), reverse=True)

    final_masks = []

    for mask in valid_masks:
        keep = True
        for existing in final_masks:
            intersection = (mask & existing).sum()
            if intersection > 0.5 * min(mask.sum(), existing.sum()):
                keep = False
                break
        if keep:
            final_masks.append(mask)

    # -----------------------------
    # SEGMENTATION DISPLAY
    # -----------------------------
    overlay = img_rgb.copy()
    for mask in final_masks:
        color = np.random.randint(0, 255, 3)
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + color * 0.5

    with col2:
        st.subheader("🧠 Segmentation")
        st.image(overlay.astype("uint8"), use_column_width=True)

    # -----------------------------
    # CALCULATIONS
    # -----------------------------
    depth_map = get_depth_map(img_rgb)

    volumes = []
    total_food_pixels = sum([m.sum() for m in final_masks])

    for mask in final_masks:
        depth_vals = depth_map[mask == 1]
        if len(depth_vals) == 0:
            volumes.append(0)
        else:
            volume = (mask.sum() ** 1.05) * (np.mean(depth_vals) + 0.1)
            volumes.append(volume)

    total_volume = sum(volumes)
    total_plate_weight = 250 + ((total_food_pixels / (img.shape[0]*img.shape[1])) * 350)

    total_cal = 0
    total_prot = 0
    total_carbs = 0
    total_fat = 0

    st.subheader("🔍 Detected Food Regions")

    cols = st.columns(len(final_masks))

    for i, mask in enumerate(final_masks):

        weight = ((volumes[i] + 1e-6) / (total_volume + 1e-6)) * total_plate_weight

        ys, xs = np.where(mask == 1)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped = img_rgb[y_min:y_max, x_min:x_max]

        input_tensor = classification_transform(cropped).unsqueeze(0).to(device)

        with torch.no_grad():
            output = effnet_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        dish = class_names[pred.item()].lower().strip()

        nutrition_row = nutrition_df[nutrition_df[food_col] == dish]

        if len(nutrition_row) != 0:
            row = nutrition_row.iloc[0]

            cal = (row[calories_per_100g] * weight) / 100
            prot = (row[protein_per_100g] * weight) / 100
            carbs = (row[carbs_per_100g] * weight) / 100
            fat = (row[fat_per_100g] * weight) / 100
        else:
            cal, prot, carbs, fat = 0, 0, 0, 0

        total_cal += cal
        total_prot += prot
        total_carbs += carbs
        total_fat += fat

        with cols[i]:
            st.image(mask * 255, caption=f"{dish.title()} ({conf.item():.2f})")
            st.markdown(f"""
            **Weight:** {weight:.1f} g  
            **Calories:** {cal:.1f} kcal  

            **Protein:** {prot:.1f} g  
            **Carbs:** {carbs:.1f} g  
            **Fat:** {fat:.1f} g  
            """)

    st.success(f"🔥 Total Calories: {total_cal:.1f} kcal")

    st.subheader("🥧 Macronutrient Distribution")

    fig, ax = plt.subplots()
    ax.pie(
        [total_prot, total_carbs, total_fat],
        labels=["Protein", "Carbs", "Fat"],
        autopct='%1.1f%%'
    )
    st.pyplot(fig)