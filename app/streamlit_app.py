import json
import os
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficient
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet
from datetime import datetime
import uuid
import difflib

# ----------------------
# Config & Paths
# ----------------------
BASE_DIR = Path('c:/Users/LQO/food_predictor')
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
NUTRITION_CSV = DATA_DIR / 'Nutrition.csv'
HISTORY_CSV = DATA_DIR / 'prediction_history.csv'
IMG_SIZE = (224, 224)

# Normalization & alias mapping (extend as needed)
ALIAS_MAP = {
    # spelling variations
    'telor': 'telur',
    'mie': 'mi',
    'nasi_goreng': 'nasi_goreng',
    'ayam_goreng': 'ayam_goreng',
    'telur_balado': 'telur_balado',
    'telor_balado': 'telur_balado',
    'telor_dadar': 'telur_dadar',
    'bakso_sapi': 'bakso',
    'bakso_ikan': 'bakso',
}

# ----------------------
# Caching Loaders
# ----------------------
@st.cache_resource
def load_best_model():
    model_path = MODELS_DIR / 'best_model.keras'
    # allow Lambda layer deserialization from our trusted local model
    model = keras.models.load_model(model_path.as_posix(), safe_mode=False)
    return model

@st.cache_data
def load_class_names():
    with open(MODELS_DIR / 'class_names.json', 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    return class_names

def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    # keep alnum and underscore only
    return ''.join(ch for ch in s if ch.isalnum() or ch == '_')

def apply_alias(s: str) -> tuple[str, bool]:
    """Return possibly remapped name using ALIAS_MAP. Outputs (name, alias_used)."""
    raw_norm = normalize_name(s)
    # direct raw -> target
    if s in ALIAS_MAP:
        return normalize_name(ALIAS_MAP[s]), True
    # normalized key
    if raw_norm in ALIAS_MAP:
        return normalize_name(ALIAS_MAP[raw_norm]), True
    # handle generic base like 'telor' -> 'telur'
    for k, v in ALIAS_MAP.items():
        if raw_norm == normalize_name(k):
            return normalize_name(v), True
    return raw_norm, False

@st.cache_data
def load_nutrition():
    """Return normalized-name -> macro dict mapping from Nutrition.csv."""
    df = pd.read_csv(NUTRITION_CSV)
    mapping = {normalize_name(row['food_name']): {
        'calories': float(row['calories']),
        'protein': float(row['protein']),
        'fat': float(row['fat']),
        'carbs': float(row['carbs'])
    } for _, row in df.iterrows()}
    return mapping

@st.cache_data
def load_nutrition_names():
    """Return display names list and display->normalized mapping for UI selections."""
    df = pd.read_csv(NUTRITION_CSV)
    display_names = [str(x) for x in df['food_name'].tolist()]
    name_to_norm = {name: normalize_name(name) for name in display_names}
    return display_names, name_to_norm

# ----------------------
# Helpers
# ----------------------

def get_preprocess_from_model(model):
    name = getattr(model, 'name', '')
    if 'efficientnetb0' in name:
        return preprocess_efficient, 'efficientnetb0'
    if 'mobilenetv2' in name:
        return preprocess_mobilenet, 'mobilenetv2'
    # default to resnet
    return preprocess_resnet, 'resnet50v2'


def load_image(image_file_or_bytes):
    if isinstance(image_file_or_bytes, (str, Path)):
        img = Image.open(image_file_or_bytes).convert('RGB')
    elif isinstance(image_file_or_bytes, BytesIO):
        img = Image.open(image_file_or_bytes).convert('RGB')
    else:
        # UploadedFile from streamlit
        img = Image.open(image_file_or_bytes).convert('RGB')
    return img


def prepare_tensor(img: Image.Image, img_size=(224, 224)):
    img = img.resize(img_size)
    arr = np.array(img).astype('float32')
    x = np.expand_dims(arr, 0)
    return x


def find_sample_image_for_label(label: str) -> Path | None:
    """Find one example image locally for a given class label (normalized/aliased)."""
    norm = normalize_name(label)
    alias_norm, _ = apply_alias(norm)
    # candidate splits to search
    splits = ['valid', 'train', 'test']
    exts = ['*.jpg', '*.jpeg', '*.png']
    base_img_dir = DATA_DIR / 'dataset_gambar'
    for sp in splits:
        class_dir = base_img_dir / sp / alias_norm
        if class_dir.exists():
            for pattern in exts:
                matches = list(class_dir.glob(pattern))
                if matches:
                    return matches[0]
    return None

def find_sample_images_for_label(label: str, k: int = 4, debug: bool = False) -> list[Path]:
    """Return up to k example image paths for a label across valid/train/test.
    This scans class subfolders and matches by normalized name to be robust to casing/spacing differences.
    """
    norm = normalize_name(label)
    alias_norm, _ = apply_alias(norm)
    splits = ['valid', 'train', 'test']
    exts = ['*.jpg', '*.jpeg', '*.png']
    base_img_dir = DATA_DIR / 'dataset_gambar'
    out: list[Path] = []
    searched = []
    for sp in splits:
        split_dir = base_img_dir / sp
        if not split_dir.exists():
            continue
        # First try direct folder
        class_dir = split_dir / alias_norm
        searched.append(class_dir)
        candidates = []
        if class_dir.exists():
            candidates.append(class_dir)
        else:
            # Fallback: scan all class folders and match by normalized name
            for sub in split_dir.iterdir():
                if sub.is_dir() and normalize_name(sub.name) == alias_norm:
                    candidates.append(sub)
        for cand in candidates:
            for pattern in exts:
                files = sorted(cand.glob(pattern))
                out.extend(files)
                if len(out) >= k:
                    break
            if len(out) >= k:
                break
        if len(out) >= k:
            break
    if debug:
        st.caption('Debug related-image search:')
        st.code('\n'.join(str(p) for p in searched))
        st.code('\n'.join(str(p) for p in out[:k]) or 'No files found')
    return out[:k]


def predict(img: Image.Image, grams: float, custom_per100: dict | None = None):
    model = load_best_model()
    class_names = load_class_names()
    nutri_map = load_nutrition()  # normalized keys

    preprocess, arch = get_preprocess_from_model(model)
    x = prepare_tensor(img, IMG_SIZE)
    x = preprocess(x)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx]
    conf = float(probs[idx])
    norm_label = normalize_name(label)
    alias_key, alias_used = apply_alias(norm_label)
    per100 = nutri_map.get(alias_key)
    found = per100 is not None
    suggestions = []
    if not found:
        # try suggest close matches
        suggestions = difflib.get_close_matches(alias_key, list(nutri_map.keys()), n=3, cutoff=0.6)
        per100 = {'calories': 0, 'protein': 0, 'fat': 0, 'carbs': 0}
    if custom_per100 is not None:
        per100 = {**per100, **{k: float(v) for k, v in custom_per100.items() if k in per100}}
    factor = max(grams, 0.0) / 100.0
    per_serving = {k: round(v * factor, 2) for k, v in per100.items()}

    # top-3
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return {
        'label': label,
        'confidence': round(conf, 4),
        'top3': top3,
        'per_100g': per100,
        'grams': grams,
        'per_serving': per_serving,
        'nutrition_found': found,
        'lookup_key': alias_key,
        'suggestions': suggestions,
        'alias_used': alias_used,
    }

def render_per100_metrics(per100: dict, title: str = 'Nutrisi per 100g'):
    """Render nutrition per 100g as clean metric cards instead of JSON."""
    st.markdown(f"### {title}")
    cols = st.columns(4)
    cols[0].metric('Kalori/100g', f"{per100.get('calories', 0)} kcal")
    cols[1].metric('Protein/100g', f"{per100.get('protein', 0)} g")
    cols[2].metric('Lemak/100g', f"{per100.get('fat', 0)} g")
    cols[3].metric('Karbo/100g', f"{per100.get('carbs', 0)} g")

def suggest_pairing_and_timing(label: str, per_serving: dict):
    """Simple rule-based suggestions for pairing and best time to eat."""
    kcal = per_serving.get('calories', 0)
    protein = per_serving.get('protein', 0)
    fat = per_serving.get('fat', 0)
    carbs = per_serving.get('carbs', 0)

    # Pairing suggestions by dish name (customize as needed)
    pairing_map = {
        'nasi_goreng': ['telur mata sapi', 'acar mentimun', 'teh tawar hangat'],
        'ayam_goreng': ['lalapan segar', 'sambal', 'nasi merah'],
        'bakso': ['sawi hijau', 'bihun', 'cabai rawit'],
        'rendang': ['nasi merah', 'sayur asem', 'teh tawar'],
        'sate': ['lontong', 'timun', 'bawang merah'],
        'soto': ['nasi putih sedikit', 'jeruk nipis', 'sambal'],
    }
    pairing = pairing_map.get(label, ['sayur hijau', 'buah segar', 'air putih'])

    # Timing: simple rules by macro profile
    now_hour = datetime.now().hour
    if carbs >= 25 and kcal >= 250:
        timing = 'Siang atau sebelum olahraga (butuh energi)'
    elif protein >= 15 and fat <= 12:
        timing = 'Siang/malam — cocok untuk pemulihan otot'
    elif kcal <= 160:
        timing = 'Pagi atau snack sore — ringan'
    else:
        timing = 'Fleksibel; sesuaikan kebutuhan harian'

    # If it's late night
    if now_hour >= 21:
        timing += ' | Catatan: Malam hari, pilih porsi lebih kecil.'

    return pairing, timing

def ensure_session_state():
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'history' not in st.session_state:
        st.session_state.history = []  # list of dicts
    if 'last_cam' not in st.session_state:
        st.session_state.last_cam = None

def add_history(entry: dict):
    # add to session
    st.session_state.history.insert(0, entry)
    # also persist to CSV (append)
    try:
        HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([entry])
        header = not HISTORY_CSV.exists()
        df.to_csv(HISTORY_CSV, mode='a', index=False, header=header)
    except Exception as e:
        st.warning(f"Tidak bisa menyimpan riwayat ke CSV: {e}")

def render_history():
    if not st.session_state.history and HISTORY_CSV.exists():
        try:
            df = pd.read_csv(HISTORY_CSV)
            st.session_state.history = df.to_dict('records')[::-1]
        except Exception:
            pass
    if not st.session_state.history:
        st.info('Belum ada riwayat.')
        return
    for h in st.session_state.history:
        with st.expander(f"{h.get('timestamp','')} · {h.get('label','?')} ({h.get('confidence',0):.2f}) · {int(h.get('grams',0))}g"):
            cols = st.columns(4)
            cols[0].metric('Kalori', f"{h['per_serving_calories']} kcal")
            cols[1].metric('Protein', f"{h['per_serving_protein']} g")
            cols[2].metric('Lemak', f"{h['per_serving_fat']} g")
            cols[3].metric('Karbo', f"{h['per_serving_carbs']} g")
            st.write('Saran pairing:', ', '.join(h.get('pairing', [])))
            st.write('Waktu disarankan:', h.get('timing', '-'))

def apply_theme(theme: str):
    """Inject bright theme CSS overrides based on selection."""
    themes = {
        'Ocean': {
            'gradient': 'linear-gradient(90deg,#06b6d4,#60a5fa,#22d3ee)',
            'button': '#2563eb',
            'button_hover': '#1d4ed8',
            'progress': '#0ea5e9',
            'accent_bg': '#ecfeff'
        },
        'Sunrise': {
            'gradient': 'linear-gradient(90deg,#f97316,#facc15,#a7f3d0)',
            'button': '#f97316',
            'button_hover': '#ea580c',
            'progress': '#f59e0b',
            'accent_bg': '#fff7ed'
        },
        'Mint': {
            'gradient': 'linear-gradient(90deg,#34d399,#a7f3d0,#fef3c7)',
            'button': '#10b981',
            'button_hover': '#059669',
            'progress': '#34d399',
            'accent_bg': '#ecfdf5'
        },
        'Grape': {
            'gradient': 'linear-gradient(90deg,#a78bfa,#f0abfc,#fde68a)',
            'button': '#7c3aed',
            'button_hover': '#6d28d9',
            'progress': '#a78bfa',
            'accent_bg': '#faf5ff'
        }
    }
    cfg = themes.get(theme, themes['Ocean'])
    st.markdown(f"""
    <style>
    .topbar {{ background: {cfg['gradient']} !important; }}
    .stButton > button{{ background: {cfg['button']} !important; }}
    .stButton > button:hover{{ background: {cfg['button_hover']} !important; }}
    .stProgress > div > div{{ background: {cfg['progress']} !important; }}
    .stMetric{{ background: {cfg['accent_bg']} !important; }}
    .stTabs [role="tab"][aria-selected="true"]{{ background: {cfg['accent_bg']} !important; }}
    </style>
    """, unsafe_allow_html=True)

# ----------------------
# UI
# ----------------------
st.set_page_config(page_title='Food Predictor + Nutrition', page_icon='🍽️', layout='wide')
ensure_session_state()

# Subtle theming tweaks
st.markdown(
    """
    <style>
    /* Layout width */
    .main .block-container{max-width:1100px}

    /* Force bright theme backgrounds */
    .stApp, .main, .block-container{
        background: #ffffff !important;
        color: #111827 !important; /* slate-900 */
    }
    /* Make most textual elements black */
    .main *:not(button){ color: #111827 !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, li, strong, em{ color:#111827 !important; }
    /* Form labels */
    .stSelectbox label, .stNumberInput label, .stCheckbox label, label{ color:#111827 !important; }
    /* Camera widget label */
    label[for^="cam_widget"]{ color:#111827 !important; }
    /* Camera widget 'Take Photo' button/text to white */
    div[data-testid="stCameraInput"] button, 
    div[data-testid="stCameraInput"] button * {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"], .stSidebar{
        background: #f8fafc !important; /* slate-50 */
    }

    /* Cards & containers */
    .stMetric {background: #f1f5f9; border-radius: 10px; padding: 8px; border: 1px solid #e2e8f0;}
    /* Force metric text to black */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"]{
        color: #111827 !important; /* black-ish */
    }
    .stExpander, .st-emotion-cache-17lntkn, .st-emotion-cache-1v0mbdj{
        background: #ffffff !important; border-radius: 10px; border: 1px solid #e5e7eb;
    }

    /* Topbar with brighter gradient */
    .topbar {background: linear-gradient(90deg,#fde68a,#a7f3d0,#bfdbfe); padding: 14px 16px; border-radius: 12px; border: 1px solid #e5e7eb;}

    /* Buttons */
    .stButton > button{
        background: #2563eb !important; /* blue-600 */
        color: white !important;
        border: none; border-radius: 8px; padding: 0.6rem 1rem;
    }
    .stButton > button:hover{background:#1d4ed8 !important}

    /* Progress bars */
    .stProgress > div > div{background: #10b981 !important} /* emerald-500 */
    .stProgress *{ color:#111827 !important; }

    /* Inputs */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] input{
        background: #ffffff !important; color: #111827 !important; border-radius: 8px;
    }
    .stNumberInput, .stSelectbox{border-radius: 8px}
    /* JSON/Code outputs */
    pre, code{ color:#111827 !important; }

    /* Tabs: force label text to black and remove blue background */
    .stTabs [role="tab"]{
        color: #111827 !important;
        background: transparent !important;
        border-radius: 6px;
    }
    .stTabs [role="tab"] p{
        color: #111827 !important;
    }
    .stTabs [role="tab"][aria-selected="true"]{
        color: #111827 !important;
        background: #f8fafc !important;
    }

    /* File uploader: bright background & dark text */
    div[data-testid="stFileUploaderDropzone"]{
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
    }
    div[data-testid="stFileUploader"] *{ color: #111827 !important; }
    div[data-testid="stFileUploader"] button{ color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="topbar"><h2>🍽️ Food Predictor + Nutrition Tracker</h2><p>Upload foto atau gunakan kamera. Atur gram porsi — nutrisi otomatis dari Nutrition.csv. Dapatkan saran pairing & waktu terbaik.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Settings')
    grams = st.number_input('Berat porsi (gram)', min_value=1.0, max_value=1200.0, value=100.0, step=1.0)
    # Custom per-100g dimatikan: nutrisi mengikuti CSV, hanya gram yang dapat diubah
    st.markdown('---')
    # Info path disembunyikan agar UI lebih bersih

tabs = st.tabs(["Kalkulator Nutrisi", "Upload Gambar", "Kamera Realtime", "Riwayat"])

with tabs[0]:
    st.subheader('Kalkulator Nutrisi (otomatis dari Nutrition.csv)')
    nutri_map = load_nutrition()  # normalized -> macros
    display_names, name_to_norm = load_nutrition_names()
    foods = sorted(display_names)

    # If there is a last prediction, offer quick select
    last_label = None
    if st.session_state.history:
        last_label = st.session_state.history[0].get('label')

    col_sel, col_gram = st.columns([2,1])
    with col_sel:
        # try to pick last label if present (match by display or normalized)
        try:
            default_index = foods.index(last_label) if last_label in foods else 0
        except ValueError:
            # try normalized
            try:
                norm_last = normalize_name(last_label) if last_label else None
                if norm_last:
                    # find display corresponding to normalized
                    for dn in foods:
                        if name_to_norm.get(dn) == norm_last:
                            default_index = foods.index(dn)
                            break
                    else:
                        default_index = 0
                else:
                    default_index = 0
            except Exception:
                default_index = 0
        food_choice = st.selectbox('Pilih makanan dari CSV', options=foods, index=default_index)
    with col_gram:
        grams_calc = st.number_input('Berat (gram)', min_value=1.0, max_value=2000.0, value=100.0, step=1.0, key='grams_calc')

    per100_csv = nutri_map.get(name_to_norm.get(food_choice,''), {'calories':0,'protein':0,'fat':0,'carbs':0})
    render_per100_metrics(per100_csv, title='Nutrisi per 100g (dari CSV)')

    # Tidak ada override: pakai CSV saja, skala dengan gram
    per100_used = per100_csv

    factor_calc = grams_calc / 100.0
    per_serv_calc = {k: round(v * factor_calc, 2) for k, v in per100_used.items()}

    st.markdown(f"### Nutrisi untuk {int(grams_calc)}g")
    ccols = st.columns(4)
    ccols[0].metric('Kalori', f"{per_serv_calc['calories']} kcal")
    ccols[1].metric('Protein', f"{per_serv_calc['protein']} g")
    ccols[2].metric('Lemak', f"{per_serv_calc['fat']} g")
    ccols[3].metric('Karbo', f"{per_serv_calc['carbs']} g")

    # Related thumbnails from dataset
    st.markdown('#### Contoh gambar terkait dari dataset')
    thumbs = find_sample_images_for_label(food_choice, k=4)
    if thumbs:
        tcols = st.columns(4)
        for i, pth in enumerate(thumbs):
            with tcols[i % 4]:
                st.image(str(pth), use_container_width=True)
    else:
        st.info('Tidak ditemukan contoh gambar lokal untuk makanan ini.')

with tabs[1]:
    st.subheader('Upload Gambar Makanan')
    up = st.file_uploader('Pilih gambar', type=['jpg', 'jpeg', 'png'])
    if up is not None:
        img = load_image(up)
        st.image(img, caption='Gambar diupload', use_container_width=True)
        if st.button('Prediksi dari Upload', type='primary'):
            result = predict(img, grams=grams)
            st.success(f"Prediksi: {result['label']} (conf: {result['confidence']:.3f})")
            st.write('Top-3 prediksi:')
            for name, p in result['top3']:
                st.progress(min(1.0, max(0.0, p)), text=f"{name} — {p:.3f}")

            render_per100_metrics(result['per_100g'])
            if not result.get('nutrition_found', True):
                st.warning('Data nutrisi tidak ditemukan untuk label tersebut di Nutrition.csv')
                suggestions = result.get('suggestions', [])
                if suggestions:
                    display_names, name_to_norm = load_nutrition_names()
                    rev_map = {v:k for k,v in name_to_norm.items()}
                    opts = [rev_map.get(s, s) for s in suggestions]
                    sel = st.selectbox('Pilih nama terdekat dari CSV untuk hitung nutrisi', options=opts, key='cam_fix')
                    chosen_norm = name_to_norm.get(sel, sel)
                    per100_fix = load_nutrition().get(chosen_norm, result['per_100g'])
                    factor_fix = result['grams'] / 100.0
                    per_serv_fix = {k: round(v * factor_fix, 2) for k, v in per100_fix.items()}
                    st.info('Perhitungan ulang berdasarkan pilihan kamu:')
                    colsfix = st.columns(4)
                    colsfix[0].metric('Kalori', f"{per_serv_fix['calories']} kcal")
                    colsfix[1].metric('Protein', f"{per_serv_fix['protein']} g")
                    colsfix[2].metric('Lemak', f"{per_serv_fix['fat']} g")
                    colsfix[3].metric('Karbo', f"{per_serv_fix['carbs']} g")
            st.markdown(f"### Nutrisi untuk {int(result['grams'])}g")
            cols = st.columns(4)
            cols[0].metric('Kalori', f"{result['per_serving']['calories']} kcal")
            cols[1].metric('Protein', f"{result['per_serving']['protein']} g")
            cols[2].metric('Lemak', f"{result['per_serving']['fat']} g")
            cols[3].metric('Karbo', f"{result['per_serving']['carbs']} g")

            pairing, timing = suggest_pairing_and_timing(result['label'], result['per_serving'])
            st.markdown('### Rekomendasi')
            st.write('• Bagus dimakan dengan:', ', '.join(pairing))
            st.write('• Waktu yang cocok:', timing)

            # Related thumbnails
            st.markdown('#### Contoh gambar terkait dari dataset')
            thumbs = find_sample_images_for_label(result['label'], k=4)
            if thumbs:
                tcols = st.columns(4)
                for i, pth in enumerate(thumbs):
                    with tcols[i % 4]:
                        st.image(str(pth), use_container_width=True)
            else:
                st.info('Tidak ditemukan contoh gambar lokal untuk label ini.')

            # add to history
            entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'label': result['label'],
                'confidence': result['confidence'],
                'grams': result['grams'],
                'per_serving_calories': result['per_serving']['calories'],
                'per_serving_protein': result['per_serving']['protein'],
                'per_serving_fat': result['per_serving']['fat'],
                'per_serving_carbs': result['per_serving']['carbs'],
                'pairing': pairing,
                'timing': timing,
                'source': 'upload',
            }
            add_history(entry)

with tabs[2]:
    st.subheader('Kamera Realtime')
    colA, colB = st.columns([1,1])
    with colA:
        if not st.session_state.camera_on:
            if st.button('▶️ Start Camera', type='primary'):
                st.session_state.camera_on = True
        else:
            if st.button('⏹️ Stop Camera'):
                st.session_state.camera_on = False
    with colB:
        capture = st.button('📸 Capture & Predict')

    placeholder = st.empty()
    if st.session_state.camera_on:
        cam = st.camera_input('Ambil foto', key='cam_widget')
        # Simpan frame terakhir yang valid agar bisa dipakai saat tombol Capture ditekan
        if cam is not None:
            st.session_state.last_cam = cam
        result = None
        if capture:
            if st.session_state.last_cam is None:
                st.warning('Belum ada foto dari kamera. Silakan ambil foto terlebih dahulu.')
            else:
                img = load_image(st.session_state.last_cam)
                placeholder.image(img, caption='Foto diambil', use_container_width=True)
                result = predict(img, grams=grams)
        if result is not None:
            st.success(f"Prediksi: {result['label']} (conf: {result['confidence']:.3f})")
            st.write('Top-3 prediksi:')
            for name, p in result['top3']:
                st.progress(min(1.0, max(0.0, p)), text=f"{name} — {p:.3f}")

            render_per100_metrics(result['per_100g'])
            if not result.get('nutrition_found', True):
                st.warning('Data nutrisi tidak ditemukan untuk label tersebut di Nutrition.csv')
                suggestions = result.get('suggestions', [])
                if suggestions:
                    display_names, name_to_norm = load_nutrition_names()
                    rev_map = {v:k for k,v in name_to_norm.items()}
                    opts = [rev_map.get(s, s) for s in suggestions]
                    sel = st.selectbox('Pilih nama terdekat dari CSV untuk hitung nutrisi', options=opts)
                    chosen_norm = name_to_norm.get(sel, sel)
                    per100_fix = load_nutrition().get(chosen_norm, result['per_100g'])
                    factor_fix = result['grams'] / 100.0
                    per_serv_fix = {k: round(v * factor_fix, 2) for k, v in per100_fix.items()}
                    st.info('Perhitungan ulang berdasarkan pilihan kamu:')
                    colsfix = st.columns(4)
                    colsfix[0].metric('Kalori', f"{per_serv_fix['calories']} kcal")
                    colsfix[1].metric('Protein', f"{per_serv_fix['protein']} g")
                    colsfix[2].metric('Lemak', f"{per_serv_fix['fat']} g")
                    colsfix[3].metric('Karbo', f"{per_serv_fix['carbs']} g")

            st.markdown(f"### Nutrisi untuk {int(result['grams'])}g")
            cols = st.columns(4)
            cols[0].metric('Kalori', f"{result['per_serving']['calories']} kcal")
            cols[1].metric('Protein', f"{result['per_serving']['protein']} g")
            cols[2].metric('Lemak', f"{result['per_serving']['fat']} g")
            cols[3].metric('Karbo', f"{result['per_serving']['carbs']} g")

            pairing, timing = suggest_pairing_and_timing(result['label'], result['per_serving'])
            st.markdown('### Rekomendasi')
            st.write('• Bagus dimakan dengan:', ', '.join(pairing))
            st.write('• Waktu yang cocok:', timing)

            entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'label': result['label'],
                'confidence': result['confidence'],
                'grams': result['grams'],
                'per_serving_calories': result['per_serving']['calories'],
                'per_serving_protein': result['per_serving']['protein'],
                'per_serving_fat': result['per_serving']['fat'],
                'per_serving_carbs': result['per_serving']['carbs'],
                'pairing': pairing,
                'timing': timing,
                'source': 'camera',
            }
            add_history(entry)

with tabs[3]:
    st.subheader('Riwayat Prediksi')
    act_col1, act_col2 = st.columns([1,1])
    with act_col1:
        if st.button('🔄 Muat Ulang Riwayat'):
            st.session_state.history = []
    with act_col2:
        if st.button('🧹 Bersihkan Riwayat'):
            st.session_state.history = []
            try:
                if HISTORY_CSV.exists():
                    HISTORY_CSV.unlink()
            except Exception as e:
                st.warning(f"Gagal menghapus file riwayat: {e}")
    render_history()

st.markdown('---')
st.caption('Tip: Jalankan dengan perintah: streamlit run app/streamlit_app.py')
