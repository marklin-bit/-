import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

# --- 1. CSS 樣式 ---
def local_css():
    st.markdown("""
    <style>
    h1 { font-size: 22px !important; padding-bottom: 0px !important; }
    h2 { font-size: 18px !important; }
    .stNumberInput label, .stSelectbox label, .stSlider label { font-size: 14px !important; font-weight: bold; }
    .main .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }
    .highlight-box {
        background-color: #e8f4f9; color: #00416d; padding: 12px;
        border-radius: 8px; border-left: 5px solid #007bff;
        text-align: center; margin-top: 5px;
    }
    .highlight-value { font-size: 22px !important; font-weight: bold; }
    .tolerance-text { font-size: 12px !important; color: #d9534f; margin-top: 4px; font-weight: bold; }
    .hint-box { font-size: 12px !important; color: #444; background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 獨立公差計算邏輯 (ID 與 CS 分開) ---
def get_id_strict_tol(id_val):
    if id_val <= 42.00: tol = 0.38
    elif id_val <= 130.00: tol = 0.76
    elif id_val <= 380.00: tol = 2.16
    elif id_val <= 580.00: tol = 3.18
    else: tol = id_val * 0.006
    return (tol / 3 / id_val) * 100

def get_cs_strict_tol(cs_val):
    # CS 公差通常較嚴格，依據標準線徑公差之 1/3 計算
    if cs_val <= 2.62: tol = 0.08
    elif cs_val <= 3.53: tol = 0.10
    elif cs_val <= 5.33: tol = 0.13
    elif cs_val <= 7.00: tol = 0.15
    else: tol = 0.20
    return (tol / 3 / cs_val) * 100

# --- 3. 數據與模型：獨立計算命中率 ---
@st.cache_resource
def load_and_train(cur_id, cur_cs):
    try:
        df = pd.read_csv('橡膠縮水率-結構化數據分析_硬度版.csv', encoding='cp950')
    except:
        df = pd.read_csv('橡膠縮水率-結構化數據分析_硬度版.csv', encoding='utf-8')
    
    df.columns = ['ID', 'CS', 'Hardness', 'Mat_Spec', 'ML', 'Method', 'Pressure', 'Fill', 'Target_ID', 'Target_CS']
    def clean(x):
        try:
            s = str(x).replace('%', '').strip()
            return float(s) if s not in ['#DIV/0!', 'nan', ''] else np.nan
        except: return np.nan

    for c in df.columns[df.columns != 'ML']: df[c] = df[c].apply(clean)
    df = df.dropna(subset=['Mat_Spec', 'Target_ID', 'Target_CS']).copy()
    
    features = ['Mat_Spec', 'ID', 'CS', 'Hardness', 'Pressure', 'Fill', 'Method']
    model_id = ExtraTreesRegressor(n_estimators=100, random_state=42).fit(df[features], df['Target_ID'])
    model_cs = ExtraTreesRegressor(n_estimators=100, random_state=42).fit(df[features], df['Target_CS'])
    
    # 獨立計算 ID 命中率
    id_tol = get_id_strict_tol(cur_id)
    id_preds = model_id.predict(df[features])
    id_hit = np.mean(np.abs(id_preds - df['Target_ID']) <= id_tol)
    
    # 獨立計算 CS 命中率
    cs_tol = get_cs_strict_tol(cur_cs)
    cs_preds = model_cs.predict(df[features])
    cs_hit = np.mean(np.abs(cs_preds - df['Target_CS']) <= cs_tol)
    
    avg_hit = (id_hit + cs_hit) / 2 * 100
    if avg_hit > 75: tag = "命中率 > 75%"
    elif avg_hit >= 35: tag = "命中率 35%~75%"
    else: tag = "命中率 < 35%"
    
    return model_id, model_cs, tag, len(df), id_tol, cs_tol

# --- 4. 介面配置 ---
st.set_page_config(page_title="橡膠縮水率預測", layout="wide")
local_css()

with st.sidebar:
    st.subheader("📌 基礎設計參數")
    id_in = st.number_input("設計內徑 ID (mm)", value=532.31, step=0.01)
    cs_in = st.number_input("設計線徑 CS (mm)", value=5.33, step=0.01)
    ms_in = st.number_input("試片縮率 (%)", value=3.10, step=0.1)
    hr_in = st.number_input("膠料硬度 (Shore A)", value=72.9, step=0.1)
    meth_in = st.selectbox("製造工法", options=[(1, "擠料"), (0, "塊料")], format_func=lambda x: x[1])[0]

model_id, model_cs, acc_tag, data_count, id_t, cs_t = load_and_train(id_in, cs_in)
st.title("🛞 橡膠縮水率預測系統")

fill_hint = """<div class="hint-box"><b>💡 填充率建議標準：</b><br>A. 一般膠料：95~105% (含墊料生產)<br>B. 易吸背料-塊料：80~85% (Max. 90%)<br>C. 易吸背料-擠料：85~90% (Max. 93%)</div>"""

tab1, tab2 = st.tabs(["🆕 新開模具預測模式", "🔄 ID 縮水率成型參數反推模式"])

with tab1:
    m1_c1, m1_c2, m1_c3 = st.columns([1.8, 1.2, 1.2])
    with m1_c1:
        p_in = st.slider("預計生產壓力 (kg/cm2)", 40, 150, 90, 10)
        f_in = st.slider("預計填充率 (%)", 80, 115, 95, 1)
    with m1_c2:
        st.markdown(fill_hint, unsafe_allow_html=True)
    with m1_c3:
        pred_id = model_id.predict([[ms_in, id_in, cs_in, hr_in, p_in, f_in, meth_in]])[0]
        pred_cs = model_cs.predict([[ms_in, id_in, cs_in, hr_in, p_in, f_in, meth_in]])[0]
        st.write("📈 預測縮水率 (±1% 公差參考)")
        st.markdown(f'''
            <div class="highlight-box">
                <div class="highlight-value">ID: {pred_id:.2f}% / CS: {pred_cs:.2f}%</div>
                <div class="tolerance-text">判定基準 ID: ±{id_t:.3f}% / CS: ±{cs_t:.3f}%</div>
            </div>
        ''', unsafe_allow_html=True)
        st.write("📊 命中率判斷")
        st.info(f"{acc_tag}")

with tab2:
    st.subheader("🔄 ID 縮水率參數反推模式")
    m2_c1, m2_c2, m2_c3 = st.columns([1.5, 1.2, 1.3])
    with m2_c1:
        st.write("**第一步：輸入目前實測基準**")
        p_now = st.number_input("目前機台壓力(kg)", 40, 150, 90, 10, key="p2")
        f_now = st.number_input("目前填充率(%)", 80, 115, 95, 1, key="f2")
        s_act = st.number_input("此條件實測縮率(%)", value=3.00, step=0.01)
    with m2_c2:
        st.markdown(fill_hint, unsafe_allow_html=True)
    with m2_c3:
        st.write("**第二步：設定目標**")
        mode_opt = st.radio("策略：", ["固定填充，求壓力", "固定壓力，求填充"], horizontal=True)
        s_tar = st.number_input("目標縮率 (%)", value=2.70, step=0.01)
        bias = s_act - model_id.predict([[ms_in, id_in, cs_in, hr_in, p_now, f_now, meth_in]])[0]
        if mode_opt == "固定填充，求壓力":
            p_range = np.linspace(40, 150, 111)
            best = p_range[np.argmin([abs(model_id.predict([[ms_in, id_in, cs_in, hr_in, p, f_now, meth_in]])[0] + bias - s_tar) for p in p_range])]
            st.success(f"結論：建議壓力調整至 **{best:.1f} kg/cm2**")
        else:
            f_range = np.linspace(80, 115, 36)
            best = f_range[np.argmin([abs(model_id.predict([[ms_in, id_in, cs_in, hr_in, p_now, f, meth_in]])[0] + bias - s_tar) for f in f_range])]
            st.success(f"結論：建議填充率調整至 **{best:.1f} %**")

st.divider()
with st.expander("📖 查看完整系統說明與聲明", expanded=True):
    st.markdown(f"""
    1. **關於命中率**：反映預測值信心度。
    2. **動態模擬**：協助製程補償判斷。
    3. **低命中率警告**：請優先參考原始數據，命中率若低於35%則不建議參考。
    4. **數據筆數**：基於 **{data_count}** 筆數據。
    """)
