"""
PlantMD — Plant Disease Detection
Run: streamlit run streamlit_app.py
Needs: plant_disease_efficientnet.keras, class_names.json
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantMD",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Disease knowledge base (all 38 PlantVillage classes) ──────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "description": "A fungal disease caused by Venturia inaequalis. Creates dark, olive-green to brown scabby lesions on leaves and fruit, reducing yield and marketability.",
        "treatment": "Apply fungicides (captan or myclobutanil) early in the season. Remove and destroy infected leaves. Prune trees for better air circulation.",
    },
    "Apple___Black_rot": {
        "description": "Caused by the fungus Botryosphaeria obtusa. Produces circular leaf spots with purple margins, rotting fruit with concentric rings, and cankers on branches.",
        "treatment": "Remove mummified fruit and dead wood. Apply copper-based fungicides. Ensure good drainage and avoid wounding trees.",
    },
    "Apple___Cedar_apple_rust": {
        "description": "A fungal disease requiring both apple and cedar/juniper hosts. Causes bright orange-yellow spots on leaves, leading to premature defoliation.",
        "treatment": "Remove nearby juniper hosts if feasible. Apply fungicides (myclobutanil) at bud break. Plant rust-resistant apple varieties.",
    },
    "Apple___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Continue regular monitoring, proper fertilisation, and seasonal pruning to maintain plant health.",
    },
    "Blueberry___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Maintain acidic soil pH (4.5–5.5), ensure adequate drainage, and monitor regularly for early signs of disease.",
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "Caused by Podosphaera clandestina. Forms a white powdery coating on young leaves and shoots, distorting growth and reducing fruit quality.",
        "treatment": "Apply sulfur-based or potassium bicarbonate fungicides. Improve air circulation through pruning. Avoid overhead irrigation.",
    },
    "Cherry_(including_sour)___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Maintain proper spacing between trees, annual pruning, and balanced fertilisation.",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Caused by Cercospora zeae-maydis. Produces rectangular, grey-to-tan lesions parallel to leaf veins, significantly reducing photosynthesis.",
        "treatment": "Plant resistant hybrids. Apply foliar fungicides (strobilurins) at early tasselling. Rotate crops and till residue.",
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Caused by Puccinia sorghi. Creates small, circular to elongated, brick-red pustules on both leaf surfaces, affecting grain fill.",
        "treatment": "Plant resistant varieties. Apply fungicides if infection occurs before tasselling. Scout fields regularly.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Caused by Exserohilum turcicum. Produces large, cigar-shaped grey-green lesions that can cause significant yield loss in severe cases.",
        "treatment": "Use resistant hybrids. Apply triazole or strobilurin fungicides. Practice crop rotation and residue management.",
    },
    "Corn_(maize)___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Ensure adequate nitrogen levels, proper plant spacing, and timely weed management.",
    },
    "Grape___Black_rot": {
        "description": "Caused by Guignardia bidwellii. Causes tan leaf lesions with dark borders, and shrivels berries into hard, black mummies.",
        "treatment": "Remove mummified berries and infected canes. Apply fungicides (mancozeb, myclobutanil) from bud break through veraison.",
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "A complex fungal trunk disease. Causes tiger-stripe leaf patterns, berry spotting, and eventual vine decline over several years.",
        "treatment": "No cure exists. Prune during dry conditions, apply wound sealants, and remove severely affected vines to prevent spread.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Caused by Pseudocercospora vitis. Produces dark brown lesions on older leaves, leading to premature defoliation and weakening vines.",
        "treatment": "Apply copper-based fungicides. Improve canopy management for airflow. Remove and destroy fallen infected leaves.",
    },
    "Grape___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Maintain canopy management, balanced irrigation, and regular scouting during the growing season.",
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "A devastating bacterial disease spread by the Asian citrus psyllid. Causes blotchy yellowing, stunted growth, and bitter, misshapen fruit with no cure.",
        "treatment": "No cure available. Control psyllid populations with insecticides. Remove and destroy infected trees to prevent spread.",
    },
    "Peach___Bacterial_spot": {
        "description": "Caused by Xanthomonas arboricola. Creates water-soaked lesions on leaves that turn brown and fall out, giving a shot-hole appearance, and pitting on fruit.",
        "treatment": "Apply copper-based bactericides during the dormant season. Plant resistant varieties. Avoid overhead irrigation.",
    },
    "Peach___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Ensure annual thinning, balanced fertilisation, and monitor for early signs of bacterial or fungal issues.",
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Caused by Xanthomonas campestris. Produces small, water-soaked spots that enlarge and turn brown with yellow halos, affecting leaves and fruit.",
        "treatment": "Use certified disease-free seed. Apply copper bactericides preventively. Rotate crops and avoid working in wet fields.",
    },
    "Pepper,_bell___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Maintain consistent watering, adequate calcium levels, and scout for early pest and disease pressure.",
    },
    "Potato___Early_blight": {
        "description": "Caused by Alternaria solani. Produces dark brown concentric ring lesions (target-board pattern) on older leaves, progressing upward through the canopy.",
        "treatment": "Apply fungicides (chlorothalonil, mancozeb) preventively. Remove infected foliage. Ensure adequate potassium nutrition.",
    },
    "Potato___Late_blight": {
        "description": "Caused by Phytophthora infestans — the pathogen behind the Irish Potato Famine. Creates water-soaked lesions that rapidly turn brown, destroying entire crops.",
        "treatment": "Apply fungicides (metalaxyl, cymoxanil) at first sign. Destroy infected plants immediately. Plant certified disease-free tubers.",
    },
    "Potato___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Hill plants properly, maintain consistent soil moisture, and scout for early blight and late blight symptoms regularly.",
    },
    "Raspberry___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Prune out old floricanes after harvest, maintain weed control, and ensure good drainage.",
    },
    "Soybean___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Rotate with non-host crops, scout regularly for soybean rust and aphids, and apply balanced fertilisation.",
    },
    "Squash___Powdery_mildew": {
        "description": "Caused by Podosphaera xanthii. Creates white powdery patches on leaf surfaces, reducing photosynthesis and causing premature senescence.",
        "treatment": "Apply potassium bicarbonate, neem oil, or sulfur fungicides. Improve air circulation. Plant resistant varieties when available.",
    },
    "Strawberry___Leaf_scorch": {
        "description": "Caused by Diplocarpon earlianum. Produces small, irregular purple-to-red spots that enlarge and cause leaf edges to appear scorched and brown.",
        "treatment": "Remove infected foliage after harvest. Apply fungicides (captan). Avoid overhead irrigation and ensure good plant spacing.",
    },
    "Strawberry___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Renovate beds after harvest, control runners, and apply straw mulch to reduce soil splash.",
    },
    "Tomato___Bacterial_spot": {
        "description": "Caused by Xanthomonas species. Creates small, dark, water-soaked lesions with yellow halos on leaves and raised, scabby spots on fruit.",
        "treatment": "Apply copper bactericides preventively. Use disease-free transplants. Rotate crops and avoid overhead irrigation.",
    },
    "Tomato___Early_blight": {
        "description": "Caused by Alternaria solani. Produces dark concentric ring lesions with yellow chlorotic halos on older leaves, progressing upward.",
        "treatment": "Apply fungicides (chlorothalonil, azoxystrobin). Remove lower infected leaves. Mulch to reduce soil splash.",
    },
    "Tomato___Late_blight": {
        "description": "Caused by Phytophthora infestans. Creates large, irregular, water-soaked dark lesions rapidly destroying foliage and fruit in cool, wet conditions.",
        "treatment": "Apply fungicides (metalaxyl) immediately. Remove and bag infected plant material. Do not compost infected debris.",
    },
    "Tomato___Leaf_Mold": {
        "description": "Caused by Passalora fulva. Produces pale green to yellow spots on upper leaf surfaces with olive-green to grey mould on the underside.",
        "treatment": "Improve greenhouse ventilation. Apply fungicides (mancozeb). Remove infected leaves and avoid wetting foliage.",
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Caused by Septoria lycopersici. Creates numerous small, circular spots with dark borders and light grey centres, causing severe defoliation.",
        "treatment": "Apply fungicides (chlorothalonil, copper). Remove infected lower leaves. Stake plants to improve air circulation.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Infestation by Tetranychus urticae. Causes stippled, bronzed leaves with fine webbing on the underside, thriving in hot, dry conditions.",
        "treatment": "Apply miticides or neem oil. Introduce predatory mites (Phytoseiulus persimilis). Increase humidity and avoid water stress.",
    },
    "Tomato___Target_Spot": {
        "description": "Caused by Corynespora cassiicola. Produces brown lesions with concentric rings and yellow halos on leaves, stems, and fruit.",
        "treatment": "Apply fungicides (azoxystrobin, difenoconazole). Improve airflow through pruning. Remove heavily infected plant material.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Spread by whiteflies (Bemisia tabaci). Causes upward leaf curling, yellowing, stunted growth, and severe yield loss with no direct cure.",
        "treatment": "Control whitefly populations with insecticides or reflective mulches. Remove infected plants. Plant resistant varieties.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Caused by Tomato mosaic virus (ToMV). Produces mosaic patterns of light and dark green, leaf distortion, and reduced fruit set.",
        "treatment": "No cure. Remove and destroy infected plants. Disinfect tools with bleach solution. Plant certified virus-free seeds.",
    },
    "Tomato___healthy": {
        "description": "No disease detected. The leaf appears healthy with no visible signs of infection or stress.",
        "treatment": "Maintain consistent watering, balanced fertilisation, and scout regularly for early signs of disease or pest pressure.",
    },
}

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

.stApp {
    background-color: #f5f0e8;
    background-image:
        radial-gradient(ellipse at 10% 10%, rgba(134,179,96,0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 90% 90%, rgba(101,141,74,0.10) 0%, transparent 60%);
}

.site-header { text-align: center; padding: 2.5rem 0 1.5rem; }
.site-logo {
    font-family: 'Playfair Display', serif;
    font-size: 3rem; font-weight: 600;
    color: #2d4a1e; letter-spacing: -1px; line-height: 1;
}
.site-logo span { color: #658d4a; font-style: italic; }
.site-tagline {
    font-size: 0.9rem; color: #7a8c6e; font-weight: 300;
    letter-spacing: 2px; text-transform: uppercase; margin-top: 0.4rem;
}

section[data-testid="stFileUploadDropzone"] {
    background: #f9faf4 !important;
    border: 2px dashed #b5cc96 !important;
    border-radius: 12px !important;
}

.result-wrapper {
    background: #ffffff; border: 1.5px solid #d6e4c4;
    border-radius: 16px; padding: 2rem 2.5rem; margin: 1.5rem 0;
    box-shadow: 0 4px 24px rgba(45,74,30,0.07);
}
.result-crop {
    font-family: 'Playfair Display', serif; font-size: 1rem;
    color: #7a8c6e; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.2rem;
}
.result-disease {
    font-family: 'Playfair Display', serif; font-size: 2rem;
    font-weight: 600; color: #2d4a1e; line-height: 1.2; margin-bottom: 0.75rem;
}
.result-disease.healthy { color: #4a7c2f; }

.confidence-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; }
.confidence-pill {
    background: #edf5e1; color: #3d6b25; font-size: 1.4rem; font-weight: 700;
    padding: 0.3rem 1rem; border-radius: 999px;
    border: 1.5px solid #b5cc96; font-family: 'Lato', sans-serif;
}
.confidence-pill.low { background: #fdf3e7; color: #a0522d; border-color: #e8c99a; }

.status-badge {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 0.25rem 0.8rem; border-radius: 999px;
}
.badge-healthy  { background: #edf5e1; color: #3d6b25; border: 1px solid #b5cc96; }
.badge-disease  { background: #fdf3e7; color: #a0522d; border: 1px solid #e8c99a; }

.info-divider { border: none; border-top: 1px solid #e8eed8; margin: 1.25rem 0; }
.info-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #9aaa88; margin-bottom: 0.4rem;
}
.info-text { font-size: 0.95rem; color: #3d4d35; line-height: 1.65; font-weight: 300; }

.treatment-box {
    background: #f2f7ea; border-left: 3px solid #7ab648;
    border-radius: 0 8px 8px 0; padding: 0.85rem 1.1rem; margin-top: 0.4rem;
}
.treatment-box p { font-size: 0.95rem; color: #3d4d35; line-height: 1.65; margin: 0; font-weight: 300; }

.low-conf-warn { font-size: 0.8rem; color: #a0522d; margin-top: 0.75rem; }

.placeholder-area { text-align: center; padding: 3.5rem 0 2.5rem; color: #9aaa88; }
.placeholder-icon { font-size: 3.5rem; }
.placeholder-text { font-size: 0.9rem; margin-top: 0.75rem; font-weight: 300; letter-spacing: 0.5px; }

.site-footer {
    text-align: center; padding: 2rem 0 1rem;
    color: #9aaa88; font-size: 0.78rem; letter-spacing: 0.5px; font-weight: 300;
}

@media (max-width: 768px) {
    .site-logo { font-size: 2.2rem; }
    .result-disease { font-size: 1.5rem; }
    .result-wrapper { padding: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_classes():
    import keras
    model_path   = Path("model/plant_disease_efficientnet.keras")
    classes_path = Path("model/class_names.json")
    if not model_path.exists():
        st.error("❌ `plant_disease_efficientnet.keras` not found.")
        st.stop()
    if not classes_path.exists():
        st.error("❌ `class_names.json` not found.")
        st.stop()
    model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)
    with open(classes_path) as f:
        classes = json.load(f)
    return model, classes


# ── Helpers ────────────────────────────────────────────────────────────────
def predict(img, model, class_names):
    import keras
    arr   = keras.utils.img_to_array(img.resize((224, 224)))
    batch = np.expand_dims(arr, 0)
    probs = model.predict(batch, verbose=0)[0]
    top   = int(np.argmax(probs))
    return class_names[top], float(probs[top])

def parse_class(raw):
    parts   = raw.split("___")
    crop    = parts[0].replace("_", " ").replace("(maize)", "(Maize)")
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    return crop, disease

def get_info(raw_class):
    info = DISEASE_INFO.get(raw_class)
    if info:
        return info["description"], info["treatment"]
    return (
        "Detailed information is not available for this class.",
        "Consult a local agricultural extension officer for guidance.",
    )


# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-header">
    <div class="site-logo">Plant<span>MD</span></div>
    <div class="site-tagline">AI-powered plant disease detection</div>
</div>
""", unsafe_allow_html=True)

model, class_names = load_model_and_classes()

uploaded = st.file_uploader(
    "Upload a clear, close-up photo of a single leaf",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img, use_container_width=True, caption="Uploaded leaf")

    with col_res:
        with st.spinner("Analysing leaf…"):
            raw_class, confidence = predict(img, model, class_names)

        crop, disease   = parse_class(raw_class)
        desc, treatment = get_info(raw_class)
        is_healthy      = "healthy" in disease.lower()
        conf_pct        = confidence * 100
        low_conf        = conf_pct < 60

        badge   = '<span class="status-badge badge-healthy">✓ Healthy</span>' if is_healthy else '<span class="status-badge badge-disease">⚠ Disease Detected</span>'
        pill_cls = "confidence-pill low" if low_conf else "confidence-pill"
        dis_cls  = "result-disease healthy" if is_healthy else "result-disease"
        low_warn = "<div class='low-conf-warn'>⚠ Low confidence — retake the photo with better lighting or a closer crop.</div>" if low_conf else ""

        st.markdown(f"""
        <div class="result-wrapper">
            <div class="result-crop">{crop}</div>
            <div class="{dis_cls}">{disease}</div>
            <div class="confidence-row">
                <span class="{pill_cls}">{conf_pct:.1f}%</span>
                {badge}
            </div>
            <hr class="info-divider">
            <div class="info-label">About this condition</div>
            <div class="info-text">{desc}</div>
            <hr class="info-divider">
            <div class="info-label">Recommended treatment</div>
            <div class="treatment-box"><p>{treatment}</p></div>
            {low_warn}
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="placeholder-area">
        <div class="placeholder-icon">🍃</div>
        <div class="placeholder-text">Upload a leaf photo above to receive an instant diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="site-footer">
    PlantMD &nbsp;·&nbsp; EfficientNetV2-S &nbsp;·&nbsp; PlantVillage Dataset &nbsp;·&nbsp; 38 crop-disease classes
</div>
""", unsafe_allow_html=True)
