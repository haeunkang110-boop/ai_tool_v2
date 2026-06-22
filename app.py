import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import io

# ── 장르별 벤치마크 컨텍스트 ─────────────────────────────────
GENRE_CONTEXT = {
    "키우기 / Idle RPG": """
- 장르 특성: 자동 사냥 기반, 짧은 세션 반복 접속, 오프라인 보상 수령이 핵심 루프
- 일반적 지표 범위: D1 리텐션 40~50%, D7 20~30%, DAU/MAU 40~60%
- PUR 평균 5~10%, ARPPU 5~15만원(국내 기준)
- 주요 이탈 원인: 콘텐츠 소진, 성장 속도 둔화, 과금 압박
""",
    "MMORPG": """
- 장르 특성: 길드/파티 사회적 결속, 장시간 플레이, 경쟁 콘텐츠 중심
- 일반적 지표 범위: D1 리텐션 35~45%, 세션 시간 30~60분
- PUR 8~15%, ARPPU 10~30만원(국내 기준)
- 주요 이탈 원인: 과금 격차, 콘텐츠 부재, 길드 해체
""",
    "전략": """
- 장르 특성: 동맹 전쟁, 장기 성장, 전략적 의사결정 중심
- 일반적 지표 범위: D1 리텐션 35~45%, 세션 시간 15~30분, 높은 DAU/MAU
- PUR 5~12%, ARPPU 고래 의존도 높음
- 주요 이탈 원인: 과금 격차, 동맹 붕괴, 서버 불균형
""",
    "퍼즐": """
- 장르 특성: 짧은 세션, 높은 DAU, 광고 수익 비중 높음
- 일반적 지표 범위: D1 리텐션 35~45%, 세션 시간 5~15분
- PUR 2~5%, ARPPU 낮음, 광고 ARPDAU 중요
- 주요 이탈 원인: 난이도 급상승, 콘텐츠 소진
""",
    "캐주얼 / 하이퍼캐주얼": """
- 장르 특성: 극단적으로 짧은 세션, 대규모 유저, 광고 수익 중심
- 일반적 지표 범위: D1 리텐션 25~40%, 세션 시간 2~5분
- PUR 1~3%, 광고 수익이 주요 BM
- 주요 이탈 원인: 반복성, 과도한 광고
""",
    "슈터": """
- 장르 특성: PvP 중심, 스킬 기반 경쟁, 시즌 패스 BM
- 일반적 지표 범위: D1 리텐션 40~50%, 세션 시간 20~40분
- PUR 5~10%, 배틀패스/스킨 중심 BM
- 주요 이탈 원인: 매칭 불균형, 핵/어뷰징, 메타 변화
"""
}

STAGE_CONTEXT = {
    "SL (소프트론치)": """
- 현재 단계: 소프트론치(SL) — 제한된 국가/유저 대상 운영 중
- 주요 관심 지표: 초기 리텐션(D1/D7/D30), CPI, CPA, 초기 ARPPU, 튜토리얼 이탈률
- 이 단계의 목표: GL 전환 가능 여부 판단, 핵심 지표 벤치마크 대비 검증
- 분석 시 GL 대비 데이터가 적을 수 있음을 감안할 것
""",
    "GL (글로벌 라이브)": """
- 현재 단계: 글로벌 라이브(GL) — 전 세계 정식 서비스 중
- 주요 관심 지표: DAU 추이, 매출 안정성, 장기 리텐션, LTV, 광고 효율
- 이 단계의 목표: 매출 유지/성장, 유저 이탈 방어, 장기 운영 효율화
"""
}

ANALYSIS_CONTEXT = {
    "라이브 지표 모니터링": "DAU, MAU, 접속 빈도, 세션 시간 등 유저 활동 지표의 추이와 이상치를 중심으로 분석하세요.",
    "수익화 분석": "총매출, ARPPU, ARPDAU, PUR 등 수익 지표의 구조와 변화를 중심으로 분석하세요.",
    "리텐션 분석": "신규/복귀/이탈 유저 추이, D1/D7/D30 리텐션, DAU/MAU 비율을 중심으로 분석하세요.",
    "UA 효율 분석": "CPI, CPA, 신규 유저 유입량, 채널별 효율을 중심으로 분석하세요."
}

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="게임 지표 분석기",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 게임 지표 분석기")
st.caption("엑셀 또는 CSV 파일을 업로드하면 AI가 PM 관점의 분석 리포트를 생성합니다.")

# ── 사이드바 ─────────────────────────────────────────────────
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

    genre = st.selectbox(
        "장르",
        ["키우기 / Idle RPG", "MMORPG", "전략", "퍼즐", "캐주얼 / 하이퍼캐주얼", "슈터"]
    )
    stage = st.selectbox(
        "게임 단계",
        ["SL (소프트론치)", "GL (글로벌 라이브)"]
    )
    analysis_focus = st.selectbox(
        "분석 목적",
        ["라이브 지표 모니터링", "수익화 분석", "리텐션 분석", "UA 효율 분석"]
    )
    st.divider()
    st.markdown("**사용법**\n1. API Key 입력\n2. 옵션 선택\n3. 파일 업로드\n4. 분석 실행 클릭")

# ── 파일 업로드 ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 파일을 여기에 드래그하거나 클릭해서 업로드하세요",
    type=["xlsx", "xls", "csv"],
    help="엑셀(.xlsx, .xls) 또는 CSV 파일을 지원합니다."
)

def smart_read_excel(file):
    df_raw = pd.read_excel(file, header=None)
    for i, row in df_raw.iterrows():
        non_null = row.notna().sum()
        if non_null >= len(row) * 0.4:
            df = pd.read_excel(file, header=i)
            df = df.dropna(how='all')
            df.columns = [str(c) if not str(c).startswith('Unnamed') else f"col_{j}"
                         for j, c in enumerate(df.columns)]
            return df
    return pd.read_excel(file)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = smart_read_excel(uploaded_file)
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

    st.success(f"✅ '{uploaded_file.name}' 로드 완료 — {df.shape[0]}행 × {df.shape[1]}열")

    tab1, tab2, tab3 = st.tabs(["📋 데이터 미리보기", "📊 차트", "🤖 AI 분석"])

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
                y_options = [c for c in numeric_cols if c != x_col]
                y_cols = st.multiselect(
                    "Y축 (지표, 복수 선택 가능)",
                    y_options,
                    default=y_options[:min(3, len(y_options))]
                )
            chart_type = st.radio("차트 유형", ["라인", "바", "스캐터"], horizontal=True)

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

    with tab3:
        st.subheader("🤖 Gemini AI 분석 리포트")
        st.caption(f"**장르:** {genre} | **단계:** {stage} | **분석 목적:** {analysis_focus}")

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
아래 조건에 맞춰 게임 지표 데이터를 분석하고 리포트를 작성해 주세요.

[분석 대상 게임 정보]
- 장르: {genre}
- 운영 단계: {stage}
- 분석 목적: {analysis_focus}

[장르 벤치마크 기준]
{GENRE_CONTEXT[genre]}

[운영 단계 컨텍스트]
{STAGE_CONTEXT[stage]}

[분석 포커스]
{ANALYSIS_CONTEXT[analysis_focus]}

[중요 제약사항]
- 반드시 데이터에 명시된 수치만 근거로 사용하세요.
- 파일에 없는 정보(이벤트, 업데이트, 외부 요인 등)는 절대 추론하거나 언급하지 마세요.
- 데이터로 확인할 수 없는 원인 분석은 하지 마세요.
- 장르 벤치마크 기준과 실제 데이터 수치를 비교하여 평가해 주세요.

[리포트 구성]
1. **핵심 요약** (3줄 이내)
2. **주요 인사이트** (bullet 3~5개, 수치 직접 인용 + 벤치마크 대비 평가)
3. **리스크 / 주의 지점** (bullet 2~3개)
4. **Action Item** (각 항목당 1개씩, 총 3개)
   - [개발] 개발 사이드 관점 액션
   - [사업PM] 사업 PM 관점 액션
   - [마케팅] 마케팅 사이드 관점 액션

- Action Item 부분 중 해당 사항이 없는 항목은 생략 가능합니다.


[데이터]
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
    - 장르·단계·목적별 맞춤 AI 분석 리포트
    - Action Item 자동 생성 [개발 / 사업PM / 마케팅]
    """)
