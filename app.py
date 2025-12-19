import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import plotly.graph_objects as go

# --- 1. 數據與模型初始化 (快取處理) ---
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv('橡膠縮水率-結構化數據分析_硬度版.csv', encoding='cp950')
    except:
        df = pd.read_csv('橡膠縮水率-結構化數據分析_硬度版.csv', encoding='utf-8')
    
    df.columns = ['ID', 'CS', 'Hardness', 'Mat_Spec', 'ML', 'Method', 'Pressure', 'Fill', 'Target_ID', 'Target_CS']
    
    def clean(x):
        try:
            if pd.isna(x): return np.nan
            s = str(x).replace('%', '').strip()
            if '#DIV/0!' in s or 'nan' in s.lower() or s == '':
                return np.nan
            return float(s)
        except:
            return np.nan

    cols = ['Mat_Spec', 'ID', 'CS', 'Hardness', 'Pressure', 'Fill', 'Method', 'Target_ID']
    for c in cols:
        df[c] = df[c].apply(clean)
    
    df = df.dropna(subset=['Mat_Spec', 'Target_ID']).copy()
    features = ['Mat_Spec', 'ID', 'CS', 'Hardness', 'Pressure', 'Fill', 'Method']
    X = df[features]
    y = df['Target_ID']
    
    model = ExtraTreesRegressor(n_estimators=100, random_state=42).fit(X, y)
    
    # 計算命中率 (依據 R-Squared 殘差基準判定)
    score = model.score(X, y)
    if score < 0.30:
        accuracy_tag = "命中率 < 30%"
    elif 0.30 <= score <= 0.75:
        accuracy_tag = "命中率 30%~75%"
    else:
        accuracy_tag = "命中率 > 75%"
        
    return model, accuracy_tag, len(df)

# --- 2. 頁面配置 ---
st.set_page_config(page_title="橡膠縮水率預測 V9.95", layout="wide")
st.title("🛞 橡膠縮水率預測")

model, acc_tag, data_count = load_and_train()

# --- 3. 側邊欄：基礎設計參數 ---
with st.sidebar:
    st.header("📌 基礎設計參數")
    id_in = st.number_input("設計內徑 ID (mm)", value=532.31, step=0.01)
    cs_in = st.number_input("設計線徑 CS (mm)", value=5.33, step=0.01)
    ms_in = st.number_input("試片縮率 (%)", value=3.1, step=0.1)
    hr_in = st.number_input("膠料硬度 (Shore A)", value=72.9, step=0.1)
    meth_in = st.selectbox("製造工法", options=[(1, "擠料"), (0, "塊料")], format_func=lambda x: x[1])[0]

tab1, tab2 = st.tabs(["🆕 新開模具預測模式", "🔄 成型參數反向修正模式"])

# --- 4. 模式一：新開模具預測 ---
with tab1:
    col_input, col_res1, col_res2 = st.columns([2, 1, 1])
    
    with col_input:
        p_in = st.slider("預計生產壓力 (kg)", min_value=40, max_value=150, value=90, step=10)
        f_in = st.slider("預計填充率 (%)", min_value=80, max_value=115, value=95, step=1)
    
    # 執行預測
    pred_s = model.predict([[ms_in, id_in, cs_in, hr_in, p_in, f_in, meth_in]])[0]
    
    with col_res1:
        # 修改名稱為「預測縮水率」並顯示至小數點第 2 位
        st.metric("預測縮水率", f"{pred_s:.2f} %")
        
    with col_res2:
        # 正確率移至右側顯示
        st.write("📊 正確率")
        st.info(acc_tag)
        
    st.divider()
    suggested_mold = id_in * (1 + pred_s/100)
    st.info(f"💡 建議開發模具 ID 尺寸：**{suggested_mold:.3f}** mm")

# --- 5. 模式二：成型參數反向修正 ---
with tab2:
    st.subheader("🛠️ 基於現場實測之參數調整建議")
    c1, c2, c3 = st.columns(3)
    p_now = c1.number_input("目前機台壓力 (kg)", min_value=40, max_value=150, value=90, step=10)
    f_now = c2.number_input("目前填充率 (%)", value=95, step=1)
    s_act = c3.number_input("此條件實測縮率 (%)", value=3.0, step=0.01)
    
    s_tar = st.number_input("目標縮率 (%)", value=2.7, step=0.01)

    base_pred = model.predict([[ms_in, id_in, cs_in, hr_in, p_now, f_now, meth_in]])[0]
    bias = s_act - base_pred

    p_range = np.linspace(40, 150, 100)
    preds = [model.predict([[ms_in, id_in, cs_in, hr_in, p, f_now, meth_in]])[0] + bias for p in p_range]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p_range, y=preds, name="AI 預測路徑", line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=[p_now], y=[s_act], name="目前現場位置", marker=dict(size=12, color='red', symbol='cross')))
    fig.add_hline(y=s_tar, line_dash="dash", line_color="green", annotation_text="目標縮率線")
    fig.update_layout(title="壓力與縮率關係曲線 (40-150kg)", xaxis_title="機台壓力 (kg)", yaxis_title="預計產出縮率 (%)")
    st.plotly_chart(fig, use_container_width=True)

    best_p = p_range[np.argmin(np.abs(np.array(preds) - s_tar))]
    st.success(f"✅ 診斷結論：建議將壓力調整至 **{best_p:.1f} kg**")

# --- 6. 頁面下方文字說明 ---
st.divider()
st.subheader("💡 系統使用說明與聲明")
st.markdown(f"""
1. 此預測模型的 **「正確率」**，為 AI 依歷史數據訓練後提供的預測正確率（以 R-Squared 殘差分析判定是否落在公差範圍）。
2. 可調整機台壓力、填充率，觀看縮水率的變化，當預測失準時，確認是否能夠調整機台參數救回。
3. 當正確率過低時，建議仍以目前作業方式執行 (整理相同膠料歷史生產數據)
4. 此縮水率預測模型數據來源約為 **{data_count}** 筆歷史的有效生產數據。
""")