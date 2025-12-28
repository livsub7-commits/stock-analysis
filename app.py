import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
import os
import time  # 待機時間用に追加

# ページ設定
st.set_page_config(page_title="戦略的資産拡大プロセス V2", layout="wide")

# ==========================================
# 設定・サイドバー入力
# ==========================================
st.sidebar.header("⚙️ 設定")

# APIキー入力
api_key_input = st.sidebar.text_input("Gemini API Key", type="password")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or api_key_input

# 資金管理設定
INITIAL_CAPITAL_JPY = st.sidebar.number_input("初期資本 (円)", value=200000, step=10000)
RISK_TOLERANCE_PCT = st.sidebar.slider("リスク許容度 (%)", 1.0, 10.0, 5.0) / 100
ATR_MULTIPLIER = st.sidebar.slider("損切り幅 (ATR倍率)", 1.0, 3.0, 2.0)

# 監視銘柄
TARGETS = {
    "NVDA": "NVIDIA Corp",
    "TSLA": "Tesla Inc",
    "SPY": "S&P 500 ETF"
}

# AIクライアント設定
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.sidebar.error(f"APIキー設定エラー: {e}")
else:
    st.sidebar.warning("⚠️ Gemini APIキーが設定されていません。")

# ==========================================
# 関数定義
# ==========================================
def calculate_indicators(df):
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['High_52W'] = df['High'].rolling(window=250).max()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(window=14).mean()
    return df

def get_usd_jpy_rate():
    try:
        data = yf.Ticker("JPY=X").history(period="1d")
        if data.empty: return 150.0
        return data['Close'].iloc[-1]
    except:
        return 150.0

# ==========================================
# メイン処理
# ==========================================
st.title("🚀 戦略的資産拡大プロセス V2")
st.markdown("### 上値余地判定・資金管理ダッシュボード")

if st.button('分析を実行', type="primary"):
    with st.spinner('市場データを分析中...'):
        usd_jpy = get_usd_jpy_rate()
        st.info(f"ℹ️ 現在の為替レート: 1ドル = {usd_jpy:.2f}円")

        cols = st.columns(len(TARGETS))

        for idx, (ticker, name) in enumerate(TARGETS.items()):
            with cols[idx]:
                try:
                    # ダウンロード
                    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
                    
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    if df.empty or len(df) < 250:
                        st.error(f"{ticker}: データ不足")
                        continue

                    df = calculate_indicators(df)
                    last = df.iloc[-1]
                    
                    is_bull = last['Close'] > last['SMA_200']
                    rsi_val = last['RSI']
                    
                    trend_color = "green" if is_bull else "red"
                    trend_icon = "✅" if is_bull else "❌"
                    
                    st.subheader(f"{name} ({ticker})")
                    st.metric("現在値", f"${last['Close']:.2f}", f"RSI: {rsi_val:.1f}")
                    st.markdown(f"**トレンド:** :{trend_color}[{trend_icon} {'強気' if is_bull else '弱気'}]")

                    high_52 = last['High_52W']
                    dist_to_high_pct = (high_52 - last['Close']) / last['Close'] * 100
                    
                    if last['Close'] >= high_52 * 0.99:
                        st.success("🚀 青天井 (真空地帯)")
                    elif dist_to_high_pct >= 10.0:
                        st.info(f"✅ 余地あり (+{dist_to_high_pct:.1f}%)")
                    else:
                        st.warning(f"⚠️ 抵抗線近し (+{dist_to_high_pct:.1f}%)")

                    if is_bull:
                        stop_loss = last['Close'] - (ATR_MULTIPLIER * last['ATR'])
                        risk_per_share_jpy = (last['Close'] - stop_loss) * usd_jpy
                        allowable_risk = INITIAL_CAPITAL_JPY * RISK_TOLERANCE_PCT
                        
                        shares = 0
                        if risk_per_share_jpy > 0:
                            shares = int(allowable_risk / risk_per_share_jpy)
                            max_buy = int(INITIAL_CAPITAL_JPY / (last['Close'] * usd_jpy))
                            shares = min(shares, max_buy)
                        
                        st.markdown("---")
                        st.write("💰 **推奨ポジション**")
                        st.write(f"購入数: **{shares}株**")
                        st.caption(f"損切ライン: ${stop_loss:.2f}")
                    else:
                        st.markdown("---")
                        st.write("⛔ **エントリー対象外**")

                    # AIコメント生成（レート制限対策済み）
                    if client and is_bull:
                        # 少し待機してAPI制限(429)を回避
                        time.sleep(2) 

                        prompt = (
                            f"{name}の株価は${last['Close']:.2f}。52週高値は${high_52:.2f}。"
                            f"上値抵抗や今後の+10%上昇可能性について、プロの視点で40文字以内でコメントして。"
                        )
                        
                        # 唯一反応があったモデルを最優先
                        target_model = "gemini-2.0-flash-exp"
                        
                        try:
                            response = client.models.generate_content(
                                model=target_model,
                                contents=prompt
                            )
                            st.info(f"🤖 AI: {response.text}")
                        except Exception as e:
                            # 429エラー（使いすぎ）の場合は優しく表示
                            err_msg = str(e)
                            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                                st.warning("⚠️ AIアクセス集中: 少し時間をおいてください")
                            elif "404" in err_msg:
                                st.caption("⚠️ AIモデルが見つかりません (404)")
                            else:
                                st.caption(f"AIエラー: {err_msg}")

                except Exception as e:
                    st.error(f"システムエラー: {e}")