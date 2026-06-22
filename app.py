import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="게임 지표 분석기",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 게임 지표 분석기")
st.caption("엑셀 또는 CSV 파일을 업로드하면 AI가 PM 관점의 분석 리포트를 생성합니다.")

# ── 사이드바: API 키 입력 ────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="https://aistudio.google.com 에서 발급"
    )
    st.divider()
    st.markdown("**분석 옵션**")
    analysis_focus = st.selectbox(
        "분석 포커스",
        ["전체 요약", "리텐션 분석", "수익화(BM) 분석", "UA / 마케팅 효율"]
    )
    st.divider()
    st.markdown("**사용법**\n1. API Key 입력\n2. 파일 업로드\n3. 분석 실행 클릭")

# ── 파일 업로드 ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 파일을 여기에 드래그하거나 클릭해서 업로드하세요",
    type=["xlsx", "xls", "csv"],
    help="엑셀(.xlsx, .xls) 또는 CSV 파일을 지원합니다."
)

def smart_read_excel(file):
    """헤더가 중간에 있는 엑셀 파일도 처리"""
    df_raw = pd.read_excel(file, header=None)
    # 헤더 행 찾기: None이 아닌 값이 절반 이상인 첫 번째 행
    for i, row in df_raw.iterrows():
        non_null = row.notna().sum()
        if non_null >= len(row) * 0.4:
            df = pd.read_excel(file, header=i)
            # 이후 빈 행 제거
            df = df.dropna(how='all')
            # 컬럼명 정리 (Unnamed 제거)
            df.columns = [str(c) if not str(c).startswith('Unnamed') else f"col_{j}" 
                         for j, c in enumerate(df.columns)]
            return df
    return pd.read_excel(file)

if uploaded_file is not None:
    # ── 데이터 로드 ──────────────────────────────────────────
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = smart_read_excel(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    st.success(f"✅ '{uploaded_file.name}' 로드 완료 — {df.shape[0]}행 × {df.shape[1]}열")

    # ── 탭 구성 ──────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📋 데이터 미리보기", "📊 차트", "🤖 AI 분석"])

    # ── TAB 1: 데이터 미리보기 ───────────────────────────────
    with tab1:
        st.subheader("데이터 미리보기")
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 행 수", f"{df.shape[0]:,}")
        with col2:
            st.metric("총 열 수", df.shape[1])
        with col3:
            st.metric("결측치", int(df.isnull().sum().sum()))

        st.subheader("기초 통계")
        st.dataframe(df.describe(), use_container_width=True)

    # ── TAB 2: 차트 ──────────────────────────────────────────
    with tab2:
        st.subheader("차트 시각화")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        if not numeric_cols:
            st.warning("수치형 컬럼이 없어 차트를 그릴 수 없습니다.")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                x_col = st.selectbox("X축 (날짜/카테고리)", all_cols, index=0)
            with col_right:
                # x_col이 numeric_cols에 있으면 제외
                y_options = [c for c in numeric_cols if c != x_col]
                y_cols = st.multiselect(
                    "Y축 (지표, 복수 선택 가능)",
                    y_options,
                    default=y_options[:min(3, len(y_options))]
                )

            chart_type = st.radio(
                "차트 유형",
                ["라인", "바", "스캐터"],
                horizontal=True
            )

            if y_cols:
                try:
                    df_chart = df[[x_col] + y_cols].copy()
                    df_chart = df_chart.dropna(subset=y_cols, how='all')
                    df_melted = df_chart.melt(id_vars=x_col, var_name="지표", value_name="값")

                    if chart_type == "라인":
                        fig = px.line(df_melted, x=x_col, y="값", color="지표", markers=True)
                    elif chart_type == "바":
                        fig = px.bar(df_melted, x=x_col, y="값", color="지표", barmode="group")
                    else:
                        fig = px.scatter(df_melted, x=x_col, y="값", color="지표")

                    fig.update_layout(
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"차트 생성 오류: {e}")
            else:
                st.info("Y축 지표를 하나 이상 선택해 주세요.")

    # ── TAB 3: AI 분석 ───────────────────────────────────────
    with tab3:
        st.subheader("🤖 Gemini AI 분석 리포트")

        if not api_key:
            st.warning("사이드바에서 Gemini API Key를 먼저 입력해 주세요.")
        else:
            if st.button("🚀 AI 분석 실행", type="primary", use_container_width=True):
                data_summary = f"""
파일명: {uploaded_file.name}
행 수: {df.shape[0]}, 열 수: {df.shape[1]}

[컬럼 목록]
{', '.join(df.columns.tolist())}

[기초 통계]
{df.describe().to_string()}

[데이터 샘플 (최대 30행)]
{df.head(30).to_string()}
"""

                prompt = f"""
당신은 모바일 게임 Business PM입니다.
아래 게임 지표 데이터를 분석하고, [{analysis_focus}] 관점에서 리포트를 작성해 주세요.

[중요 제약사항]
- 반드시 데이터에 명시된 수치만 근거로 사용하세요.
- 파일에 없는 정보(이벤트, 업데이트, 외부 요인 등)는 절대 추론하거나 언급하지 마세요.
- 데이터로 확인할 수 없는 원인 분석은 하지 마세요.

리포트 구성:
1. **핵심 요약** (3줄 이내)
2. **주요 인사이트** (bullet 3~5개, 수치 직접 인용)
3. **리스크 / 주의 지점** (bullet 2~3개)
4. **Action Item** (각 항목당 1개씩, 총 3개)
   - [개발] 개발 사이드 관점 액션
   - [사업PM] 사업 PM 관점 액션
   - [마케팅] 마케팅 사이드 관점 액션

데이터:
{data_summary}

한국어로 작성해 주세요.
"""

                with st.spinner("Gemini가 데이터를 분석 중입니다..."):
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel("gemini-2.5-flash")
                        response = model.generate_content(prompt)
                        result = response.text
                        st.markdown(result)

                        st.divider()
                        st.download_button(
                            label="📥 리포트 텍스트로 저장",
                            data=result,
                            file_name="ai_analysis_report.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"❌ 오류 발생: {e}")

else:
    st.info("👆 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.markdown("""
    **지원 형식**
    - 엑셀: `.xlsx`, `.xls`  
    - CSV: `.csv`

    **분석 내용**
    - 데이터 미리보기 및 기초 통계
    - 인터랙티브 차트 (라인 / 바 / 스캐터)
    - Gemini AI 기반 PM 관점 인사이트 리포트
    - Action Item 자동 생성
    """)
