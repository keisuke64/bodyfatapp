import streamlit as st
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from google import genai
from supabase import create_client, Client

# ============================================================
# 1. ページ設定
# ============================================================
st.set_page_config(layout="wide", page_title="AIパーソナルフィットネス")

# ============================================================
# 2. カスタムCSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }

.stApp { background-color: #f6f4f0 !important; }
.stApp > header { background-color: transparent !important; }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important; color: #1a1a1a !important;
    font-weight: 400 !important; margin-bottom: 0.2rem !important;
}
h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.3rem !important; color: #1a1a1a !important;
    font-weight: 400 !important; margin-bottom: 1rem !important;
}
h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important; font-weight: 500 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    color: #999 !important; margin-bottom: 1rem !important;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.08) !important;
    margin: 1rem 0 1.5rem !important;
}

/* --- 入力フォーム --- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background-color: #fff !important;
    border: 1.5px solid rgba(0,0,0,0.12) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important; color: #1a1a1a !important;
    padding: 10px 14px !important; box-shadow: none !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2d5a3d !important; box-shadow: none !important;
}
.stTextInput label, .stTextArea label, .stNumberInput label {
    font-size: 12px !important; font-weight: 500 !important;
    color: #555 !important; font-family: 'DM Sans', sans-serif !important;
}

/* --- ボタン --- */
.stButton > button[kind="primary"] {
    background-color: #2d5a3d !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 12px 20px !important; box-shadow: none !important;
    transition: background-color 0.2s !important;
}
.stButton > button[kind="primary"]:hover { background-color: #4a8c5c !important; }

.stButton > button:not([kind="primary"]) {
    background-color: transparent !important; color: #555 !important;
    border: 1.5px solid rgba(0,0,0,0.15) !important;
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 400 !important;
    padding: 8px 20px !important; box-shadow: none !important;
    transition: all 0.2s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #555 !important; color: #1a1a1a !important;
}

/* --- メトリクスカード --- */
[data-testid="metric-container"] {
    background-color: #e8f2eb !important;
    border-radius: 12px !important; padding: 18px 20px !important;
}
[data-testid="metric-container"] label {
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    color: #2d5a3d !important;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem !important; color: #2d5a3d !important;
}

/* --- アラート --- */
.stAlert {
    border-radius: 10px !important; border: none !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 14px !important;
}

/* --- タブ --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px !important; background-color: transparent !important;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    background-color: #fff !important;
    border: 1.5px solid rgba(0,0,0,0.12) !important;
    border-radius: 50px !important; padding: 6px 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important; color: #555 !important;
}
.stTabs [aria-selected="true"] {
    background-color: #2d5a3d !important;
    border-color: #2d5a3d !important; color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* --- データフレーム --- */
.stDataFrame {
    border-radius: 10px !important; overflow: hidden !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
}

/* --- AIアドバイスボックス --- */
.ai-advice-box {
    background: #f0ede8; border-left: 3px solid #4a8c5c;
    border-radius: 0 10px 10px 0; padding: 16px 20px;
    margin-top: 12px; font-size: 14px; line-height: 1.7; color: #333;
}
.ai-advice-label {
    font-size: 11px; font-weight: 500; letter-spacing: 0.07em;
    text-transform: uppercase; color: #4a8c5c; margin-bottom: 8px;
}

/* --- 認証フォームのタブ区切り線を非表示 --- */
.auth-tab .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* --- サイドバー非表示 --- */
[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. 各種クライアント・モデルの準備
# ============================================================
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('bodyfat.csv')
    features = ['Weight', 'Height', 'Wrist', 'Abdomen']
    model = LinearRegression()
    model.fit(df[features], df['BodyFat'])
    return model

model = load_and_train_model()

try:
    ai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("Gemini APIキーの設定を確認してください。")
    st.stop()

try:
    supabase: Client = create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )
except Exception:
    st.error("Supabaseの設定（URLまたはKEY）を確認してください。")
    st.stop()

# ============================================================
# 4. セッション状態の初期化
# ============================================================
for key, default in {
    "user": None,          # Supabase Auth の user オブジェクト
    "access_token": None,  # セッショントークン
    "display_name": "",    # プロフィール上の表示名
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================
# 5. ヘルパー関数
# ============================================================
def is_logged_in() -> bool:
    return st.session_state.user is not None

def do_logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.display_name = ""
    st.rerun()

def get_display_name(user) -> str:
    """user_metadata に保存した display_name を返す。なければメールの@前を使う。"""
    meta = user.user_metadata or {}
    name = meta.get("display_name", "")
    if not name:
        name = user.email.split("@")[0]
    return name

# ============================================================
# 6. 認証画面（未ログイン時）
# ============================================================
if not is_logged_in():

    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🏋️ AIパーソナルフィットネス")
        st.markdown(
            "<p style='color:#777;font-size:14px;margin-top:-8px;margin-bottom:24px;'>"
            "AIが体の変化をパーソナルサポート</p>",
            unsafe_allow_html=True
        )

        # ログイン / サインアップ タブ
        tab_login, tab_signup = st.tabs(["ログイン", "新規登録"])

        # ---- ログインタブ ----
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            login_email    = st.text_input("メールアドレス", key="li_email",
                                           placeholder="example@email.com")
            login_password = st.text_input("パスワード", type="password", key="li_pass",
                                           placeholder="パスワードを入力")

            if st.button("ログイン", type="primary", use_container_width=True, key="btn_login"):
                if not login_email or not login_password:
                    st.warning("メールアドレスとパスワードを入力してください。")
                else:
                    try:
                        res = supabase.auth.sign_in_with_password(
                            {"email": login_email, "password": login_password}
                        )
                        st.session_state.user         = res.user
                        st.session_state.access_token = res.session.access_token
                        st.session_state.display_name = get_display_name(res.user)
                        st.rerun()
                    except Exception as e:
                        err = str(e)
                        if "Invalid login credentials" in err:
                            st.error("メールアドレスまたはパスワードが違います。")
                        else:
                            st.error(f"ログインに失敗しました: {err}")
                            st.write(type(e), vars(e) if hasattr(e, '__dict__') else "")

        # ---- サインアップタブ ----
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            su_name     = st.text_input("表示名（ニックネーム）", key="su_name",
                                        placeholder="例：たろう")
            su_email    = st.text_input("メールアドレス", key="su_email",
                                        placeholder="example@email.com")
            su_password = st.text_input("パスワード（6文字以上）", type="password", key="su_pass",
                                        placeholder="パスワードを設定")
            su_password2 = st.text_input("パスワード（確認）", type="password", key="su_pass2",
                                         placeholder="もう一度入力")

            if st.button("アカウントを作成", type="primary", use_container_width=True, key="btn_signup"):
                # バリデーション
                if not su_name or not su_email or not su_password or not su_password2:
                    st.warning("すべての項目を入力してください。")
                elif len(su_password) < 6:
                    st.warning("パスワードは6文字以上で設定してください。")
                elif su_password != su_password2:
                    st.warning("パスワードが一致しません。")
                else:
                    try:
                        res = supabase.auth.sign_up({
                            "email":    su_email,
                            "password": su_password,
                            "options": {
                                "data": {"display_name": su_name}  # user_metadataに保存
                            }
                        })

                        # メール認証なし設定の場合、sign_up直後にsessionが返る
                        if res.session:
                            st.session_state.user         = res.user
                            st.session_state.access_token = res.session.access_token
                            st.session_state.display_name = su_name
                            st.rerun()
                        else:
                            # Supabase側でメール確認が有効になっている場合のフォールバック
                            st.info("アカウントを作成しました。確認メールをご確認ください。")
                    except Exception as e:
                        err = str(e)
                        if "already registered" in err or "User already registered" in err:
                            st.error("このメールアドレスはすでに登録されています。ログインしてください。")
                        else:
                            st.error(f"登録に失敗しました: {err}")
                            st.write(type(e), vars(e) if hasattr(e, '__dict__') else "")

# ============================================================
# 7. メイン画面（ログイン後）
# ============================================================
else:
    user         = st.session_state.user
    display_name = st.session_state.display_name
    user_id      = user.id  # UUID — DBの絞り込みに使う

    # --- ヘッダー ---
    col_header, col_logout = st.columns([0.82, 0.18])
    with col_header:
        st.markdown(f"# {display_name} さんのマイページ")
        today_str = datetime.date.today().strftime("%Y年%-m月%-d日")
        st.markdown(
            f"<p style='color:#888;font-size:13px;margin-top:-12px;'>{today_str}</p>",
            unsafe_allow_html=True
        )
    with col_logout:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ログアウト", use_container_width=True):
            do_logout()

    st.divider()

    left_col, right_col = st.columns([0.5, 0.5], gap="large")

    # ---- 左カラム：入力と測定 ----
    with left_col:
        st.markdown("### 本日のデータ入力")

        c1, c2 = st.columns(2)
        with c1:
            weight_kg  = st.number_input("体重 (kg)",     min_value=30.0,  max_value=200.0, value=70.0,  step=0.1)
            height_cm  = st.number_input("身長 (cm)",     min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        with c2:
            wrist_cm   = st.number_input("手首周り (cm)", min_value=10.0,  max_value=20.0,  value=18.0,  step=0.1)
            abdomen_cm = st.number_input("腹囲 (cm)",     min_value=50.0,  max_value=150.0, value=85.0,  step=0.1)

        user_goal = st.text_area(
            "AIトレーナーへのメッセージ",
            placeholder="例：今日は筋トレを頑張りました！",
            height=100
        )

        weight_lbs    = weight_kg * 2.20462
        height_inches = height_cm / 2.54

        if st.button("測定してAIアドバイスを受け取る", type="primary", use_container_width=True):

            input_data   = [[weight_lbs, height_inches, wrist_cm, abdomen_cm]]
            predicted_bf = max(2.0, min(model.predict(input_data)[0], 50.0))
            bmi          = weight_kg / ((height_cm / 100) ** 2)

            # Supabaseへ保存（username の代わりに user_id で紐付け）
            try:
                supabase.table("fitness_history").insert({
                    "user_id":      user_id,
                    "username":     display_name,   # 表示用に残す（任意）
                    "weight_kg":    round(weight_kg, 1),
                    "bodyfat_pct":  round(predicted_bf, 1),
                    "abdomen_cm":   round(abdomen_cm, 1),
                    "user_comment": user_goal
                }).execute()
            except Exception as e:
                st.error("データベースへの保存に失敗しました。")
                st.write(e)

            # 結果表示
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 本日の測定結果")
            res1, res2 = st.columns(2)
            res1.metric(label="推定体脂肪率", value=f"{predicted_bf:.1f} %")
            res2.metric(label="BMI",          value=f"{bmi:.1f}")

            # 過去データ取得（user_id で絞り込み）
            history_context = ""
            try:
                hist = (
                    supabase.table("fitness_history")
                    .select("*")
                    .eq("user_id", user_id)
                    .order("created_at", desc=True)
                    .limit(2)
                    .execute()
                )
                if len(hist.data) > 1:
                    past = hist.data[1]
                    past_date = past["created_at"][:10]
                    history_context = (
                        f"なお、このクライアントの過去（前回）の記録は"
                        f"【日付: {past_date}, 体重: {past['weight_kg']}kg, "
                        f"体脂肪率: {past['bodyfat_pct']}%, 腹囲: {past['abdomen_cm']}cm】です。"
                        f"前回からの変化を踏まえてフィードバックしてください。"
                    )
            except Exception:
                pass

            # AIアドバイス
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### AIトレーナーからのメッセージ")
            with st.spinner("AIトレーナーがデータを分析中..."):
                prompt = f"""
                あなたは非常に優秀で親しみやすいプロのパーソナルトレーナーです。
                クライアントの「{display_name}」さんに対して、本日のデータと過去の経緯を基に、
                アドバイスと応援メッセージを届けてください。

                【本日のデータ】
                ・お名前: {display_name} さん
                ・身長: {height_cm} cm / 体重: {weight_kg} kg
                ・BMI: {bmi:.1f} / 推定体脂肪率: {predicted_bf:.1f} %
                ・腹囲: {abdomen_cm} cm

                {history_context}

                【本日のコメント】
                {user_goal if user_goal else "特になし。"}

                【出力形式】
                必ず冒頭で名前を呼んでください。Markdownを使って読みやすく装飾してください。
                """
                try:
                    response = ai_client.models.generate_content(
                        model='gemini-2.5-flash', contents=prompt
                    )
                    st.markdown(
                        f'<div class="ai-advice-box">'
                        f'<div class="ai-advice-label">🤖 AIトレーナーより</div>'
                        f'{response.text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    st.toast("✓ データをクラウドに保存しました！")
                except Exception as e:
                    st.error("AIアドバイスの生成中にエラーが発生しました。")

    # ---- 右カラム：履歴グラフ ----
    with right_col:
        st.markdown("### 推移グラフ・履歴")

        try:
            db_res = (
                supabase.table("fitness_history")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .execute()
            )

            if not db_res.data:
                st.markdown("<br>", unsafe_allow_html=True)
                st.info("まだ記録がありません。左側で測定すると、ここに履歴とグラフが蓄積されます。")
            else:
                records = []
                for item in db_res.data:
                    raw_date = str(item.get("created_at", ""))
                    date_str = raw_date[:10] if len(raw_date) >= 10 else "不明"
                    records.append({
                        "日付":         date_str,
                        "体重 (kg)":    item["weight_kg"],
                        "体脂肪率 (%)": item["bodyfat_pct"],
                        "腹囲 (cm)":    item["abdomen_cm"]
                    })
                my_history_df = pd.DataFrame(records)
                chart_data    = my_history_df.set_index("日付")

                tab1, tab2, tab3 = st.tabs(["体脂肪率 (%)", "体重 (kg)", "腹囲 (cm)"])
                with tab1:
                    st.line_chart(chart_data["体脂肪率 (%)"], color="#2d5a3d")
                with tab2:
                    st.line_chart(chart_data["体重 (kg)"],    color="#2d5a3d")
                with tab3:
                    st.line_chart(chart_data["腹囲 (cm)"],    color="#2d5a3d")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 履歴一覧")
                st.dataframe(
                    my_history_df.iloc[::-1],
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:
            st.error("履歴データの取得に失敗しました。")