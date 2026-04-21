import streamlit as st
import numpy as np
import pickle
import sqlite3
import hashlib
import os
import io
import base64
import json
from datetime import datetime
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VOGUE.AI — Fashion Recommender",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# INJECTED CSS  — editorial / luxury dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@300;400&family=Cormorant+Garamond:ital,wght@0,300;1,300&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0a !important;
    color: #e8e0d5 !important;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(180,140,90,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 80% 90%, rgba(120,80,40,0.06) 0%, transparent 60%),
        #0a0a0a !important;
}

/* hide default streamlit header/footer */
[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #2a2218 !important;
}
[data-testid="stSidebar"] * { color: #c9b99a !important; }

/* ── HEADINGS ── */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #e8d5b0 !important;
    letter-spacing: 0.02em;
}

/* ── INPUTS ── */
input[type="text"], input[type="password"] {
    background: #141414 !important;
    border: 1px solid #3a3020 !important;
    border-radius: 2px !important;
    color: #e8e0d5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 0.75rem !important;
    transition: border-color 0.2s ease;
}
input:focus {
    border-color: #b8924a !important;
    outline: none !important;
    box-shadow: 0 0 0 1px rgba(184,146,74,0.3) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #b8924a !important;
    color: #b8924a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.8rem !important;
    border-radius: 1px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    background: #b8924a !important;
    color: #0a0a0a !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #3a3020 !important;
    border-radius: 4px !important;
    background: #0f0f0f !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #b8924a !important; }

/* ── IMAGES in columns ── */
[data-testid="stImage"] img {
    border-radius: 2px;
    border: 1px solid #1e1a12;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
[data-testid="stImage"] img:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(184,146,74,0.15);
}

/* ── TAB BAR ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #2a2218 !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #7a6e5e !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.5rem !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #b8924a !important;
    border-bottom-color: #b8924a !important;
}

/* ── ALERTS / INFO ── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    border-left: 3px solid #b8924a !important;
    background: #141209 !important;
    color: #c9b99a !important;
}

/* ── DIVIDER ── */
hr { border-color: #2a2218 !important; margin: 1.5rem 0 !important; }

/* ── METRIC TILES ── */
[data-testid="stMetric"] {
    background: #111 !important;
    border: 1px solid #1e1a12 !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: #7a6e5e !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #e8d5b0 !important; font-family: 'Playfair Display', serif !important; }

/* ── HISTORY CARD ── */
.history-card {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    border: 1px solid #1e1a12;
    border-radius: 2px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    background: #0f0f0f;
    transition: border-color 0.2s;
}
.history-card:hover { border-color: #3a3020; }
.history-card img {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 1px;
    border: 1px solid #1e1a12;
}
.history-meta { flex: 1; }
.history-meta .ts {
    font-size: 0.68rem;
    color: #5a5040;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.history-meta .recs {
    font-size: 0.78rem;
    color: #9a8e7e;
    margin-top: 0.3rem;
}

/* ── BRAND HEADER ── */
.brand-header {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #1e1a12;
    margin-bottom: 2.5rem;
}
.brand-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    color: #e8d5b0;
    letter-spacing: 0.3em;
    font-weight: 400;
    line-height: 1;
}
.brand-subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #7a6e5e;
    letter-spacing: 0.2em;
    margin-top: 0.5rem;
}
.brand-line {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #b8924a, transparent);
    margin: 1.2rem auto 0;
}

/* ── AUTH BOX ── */
.auth-wrap {
    max-width: 420px;
    margin: 4rem auto;
    border: 1px solid #2a2218;
    border-radius: 2px;
    padding: 2.5rem 2rem;
    background: #0f0f0f;
}
.auth-label {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7a6e5e;
    margin-bottom: 0.3rem;
    display: block;
}

/* ── USER PILL ── */
.user-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #b8924a;
    border: 1px solid #3a3020;
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border: 1px solid #1e1a12 !important;
    border-radius: 2px !important;
    background: #0f0f0f !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #3a3020; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #b8924a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATABASE  — SQLite
# ─────────────────────────────────────────────
DB_PATH = "fashion_users.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT UNIQUE NOT NULL,
                password  TEXT NOT NULL,
                created   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                timestamp     TEXT NOT NULL,
                uploaded_img  BLOB,
                rec_filenames TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        """)

init_db()

# ─────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────
def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username: str, password: str) -> tuple[bool, str]:
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password, created) VALUES (?, ?, ?)",
                (username.strip(), hash_pw(password), datetime.now().isoformat())
            )
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "Username already taken."

def login_user(username: str, password: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username.strip(), hash_pw(password))
        ).fetchone()
    return dict(row) if row else None

def get_user_stats(user_id: int) -> dict:
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) as c FROM history WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
        last = conn.execute(
            "SELECT timestamp FROM history WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
        ).fetchone()
    return {"total": count, "last": last["timestamp"][:10] if last else "—"}

def save_history(user_id: int, img: Image.Image, rec_filenames: list[str]):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    blob = buf.getvalue()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO history (user_id, timestamp, uploaded_img, rec_filenames) VALUES (?,?,?,?)",
            (user_id, datetime.now().isoformat(), blob, json.dumps(rec_filenames))
        )

def get_history(user_id: int, limit: int = 20) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]

# ─────────────────────────────────────────────
# MODEL  (cached, CPU / GPU)
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t).squeeze().cpu().numpy()
    return out / (np.linalg.norm(out) + 1e-8)

def recommend(feat: np.ndarray, feature_list: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(6, len(feature_list)), metric="euclidean")
    nn.fit(feature_list)
    dists, indices = nn.kneighbors([feat])
    return indices[0][1:]

# ─────────────────────────────────────────────
# LOAD EMBEDDINGS
# ─────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    if not os.path.exists("embeddings.pkl") or not os.path.exists("filenames.pkl"):
        return None, None
    feats = np.array(pickle.load(open("embeddings.pkl", "rb")))
    fnames = pickle.load(open("filenames.pkl", "rb"))
    return feats, fnames

features, filenames = load_embeddings()

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None      # dict with id, username
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"

# ─────────────────────────────────────────────
# BRAND HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div class="brand-title">VOGUE·AI</div>
    <div class="brand-subtitle">Visual fashion intelligence</div>
    <div class="brand-line"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTH  ── Login / Register
# ─────────────────────────────────────────────
def auth_screen():
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        tab_login, tab_reg = st.tabs(["Sign In", "Register"])

        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="your username")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("ENTER →", key="btn_login", use_container_width=True):
                if not username or not password:
                    st.error("Fill in both fields.")
                else:
                    u = login_user(username, password)
                    if u:
                        st.session_state.user = u
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")

        with tab_reg:
            st.markdown("<br>", unsafe_allow_html=True)
            new_user = st.text_input("Choose username", key="reg_user", placeholder="your username")
            new_pw   = st.text_input("Choose password", type="password", key="reg_pw", placeholder="min 6 chars")
            new_pw2  = st.text_input("Confirm password", type="password", key="reg_pw2", placeholder="repeat")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("CREATE ACCOUNT →", key="btn_reg", use_container_width=True):
                if len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords don't match.")
                elif not new_user.strip():
                    st.error("Enter a username.")
                else:
                    ok, msg = register_user(new_user, new_pw)
                    if ok:
                        st.success(msg + " Sign in now.")
                    else:
                        st.error(msg)

# ─────────────────────────────────────────────
# MAIN APP  ── post-login
# ─────────────────────────────────────────────
def main_app():
    user = st.session_state.user

    # ── Top bar
    top_l, top_r = st.columns([7, 1])
    with top_l:
        st.markdown(
            f'<span class="user-pill">⬡ {user["username"]}</span>',
            unsafe_allow_html=True
        )
    with top_r:
        if st.button("Sign out", key="logout"):
            st.session_state.user = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs
    tab_rec, tab_hist = st.tabs(["Recommend", "My History"])

    # ─── TAB 1: Recommender ──────────────────
    with tab_rec:
        if features is None:
            st.error("⚠️  embeddings.pkl / filenames.pkl not found. Run `build_torch.py` first.")
            return

        st.markdown("#### Upload a garment image")
        uploaded = st.file_uploader(
            "PNG · JPG · JPEG", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
        )

        if uploaded:
            try:
                img = Image.open(uploaded).convert("RGB")

                img_col, _, info_col = st.columns([1.2, 0.2, 2])
                with img_col:
                    st.image(img, caption="Your upload", use_container_width=True)
                with info_col:
                    st.markdown("#### Finding similar pieces…")
                    with st.spinner("Extracting visual features"):
                        feat = extract_features(img)
                        indices = recommend(feat, features)

                    rec_names = [filenames[i] for i in indices]
                    st.success(f"Found {len(rec_names)} similar items")
                    st.markdown(
                        "<span style='font-size:0.75rem;color:#5a5040;letter-spacing:0.1em;text-transform:uppercase;'>"
                        "Results based on ResNet-50 visual embeddings</span>",
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                st.markdown("#### Similar Items")
                cols = st.columns(len(rec_names))
                for i, col in enumerate(cols):
                    with col:
                        st.image(rec_names[i], use_container_width=True)

                # save to history
                save_history(user["id"], img, rec_names)
                st.caption("✓ Saved to your history")

            except Exception as e:
                st.error(f"Error: {e}")

    # ─── TAB 2: History ─────────────────────
    with tab_hist:
        stats = get_user_stats(user["id"])

        m1, m2, _ = st.columns([1, 1, 3])
        with m1:
            st.metric("Total Searches", stats["total"])
        with m2:
            st.metric("Last Search", stats["last"])

        st.markdown("---")

        history = get_history(user["id"])
        if not history:
            st.info("No searches yet. Upload an image to get started.")
        else:
            for entry in history:
                ts = entry["timestamp"][:19].replace("T", "  ")
                recs = json.loads(entry["rec_filenames"]) if entry["rec_filenames"] else []

                with st.expander(f"🕐  {ts}  —  {len(recs)} recommendations", expanded=False):
                    h_l, h_r = st.columns([1, 3])
                    with h_l:
                        if entry["uploaded_img"]:
                            thumb = Image.open(io.BytesIO(entry["uploaded_img"]))
                            st.image(thumb, caption="Uploaded", use_container_width=True)
                    with h_r:
                        if recs:
                            rec_cols = st.columns(min(5, len(recs)))
                            for j, rc in enumerate(rec_cols):
                                with rc:
                                    if j < len(recs) and os.path.exists(recs[j]):
                                        st.image(recs[j], use_container_width=True)
                                    else:
                                        st.caption("—")
                        else:
                            st.caption("No recommendation data.")


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.user is None:
    auth_screen()
else:
    main_app()