import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
from streamlit_calendar import calendar
import os

from crawler import crawl_naver_view_titles
from rag_index import create_faiss_index
from rag_search import rag_query
from korea_dart_loader import get_corp_list, get_recent_disclosures , analyze_disclosure_with_rag

#########################################
# 1) 미국 상장주 리스트 로드 & 클린 필터링
#########################################
@st.cache_data
def load_clean_us_symbols():
    """
    ✅ S&P500 + NASDAQ/NYSE 전체 심볼 로드
    ✅ ETF, Test Issue 제거
    ✅ BRK.A → BRK-A 변환
    ✅ 캐싱 후 재실행 시 빠르게 로드
    """
    cache_path = "data/clean_us_symbols.csv"
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return df["Symbol"].tolist()

    # --- 1) S&P500 심볼 ---
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(sp500_url)[0]
    sp500_symbols = sp500_table["Symbol"].tolist()

    # --- 2) NASDAQ/NYSE/AMEX ---
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    other_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"

    try:
        df1 = pd.read_csv(nasdaq_url, sep="|")
        df2 = pd.read_csv(other_url, sep="|")

        # ✅ ETF / Test Issue 제거
        df1_clean = df1[(df1["ETF"] == "N") & (df1["Test Issue"] == "N")]
        df2_clean = df2[(df2["Test Issue"] == "N")]

        # ✅ Symbol 컬럼 합치기
        all_symbols_raw = pd.concat([
            df1_clean["Symbol"],
            df2_clean["ACT Symbol"]
        ]).dropna().unique().tolist()
    except Exception as e:
        st.warning(f"⚠️ NASDAQ/NYSE 리스트 로드 실패 → {e}")
        all_symbols_raw = []

    # --- 3) yfinance 호환 심볼 변환 (. → -)
    def normalize_symbol(sym):
        return sym.replace(".", "-").strip()

    sp500_symbols = [normalize_symbol(s) for s in sp500_symbols]
    all_symbols_raw = [normalize_symbol(s) for s in all_symbols_raw]

    # --- 4) 통합 & 중복 제거 ---
    combined = sorted(set(sp500_symbols + all_symbols_raw))

    # --- 5) 이상한 심볼 제거 ---
    valid_symbols = [
        s for s in combined
        if (s.isalnum() or "-" in s) and 1 < len(s) <= 6
    ]

    # --- 6) 캐싱 ---
    os.makedirs("data", exist_ok=True)
    pd.DataFrame({"Symbol": valid_symbols}).to_csv(cache_path, index=False)

    return valid_symbols

#########################################
# 2) yfinance 회사명 가져오기
#########################################
@st.cache_data
def fetch_company_names(symbols):
    """
    ✅ 선택된 심볼의 회사명(Long Name) 가져오기
    """
    result = {}
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            long_name = info.get("longName") or info.get("shortName") or "Unknown Company"
            result[sym] = long_name
        except Exception:
            result[sym] = "Unknown Company"
    return result

#########################################
# 3) yfinance 어닝 일정 가져오기
#########################################
def get_earnings_calendar(symbols):
    events = []
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        info = stock.calendar

        if hasattr(info, "index") and "Earnings Date" in info.index:
            earnings_date_raw = info.loc["Earnings Date"][0]
            eps_estimate = (
                info.loc["Earnings Average"][0]
                if "Earnings Average" in info.index else "N/A"
            )
        elif isinstance(info, dict) and "Earnings Date" in info:
            earnings_date_raw = info["Earnings Date"]
            eps_estimate = info.get("Earnings Average", "N/A")
        else:
            continue

        if isinstance(earnings_date_raw, list) and len(earnings_date_raw) > 0:
            earnings_date = earnings_date_raw[0]
        else:
            earnings_date = earnings_date_raw

        if hasattr(earnings_date, "strftime"):
            earnings_date = earnings_date.strftime("%Y-%m-%d")
        elif not isinstance(earnings_date, str):
            earnings_date = str(earnings_date)

        events.append({
            "title": f"{symbol} 어닝콜 (EPS {eps_estimate})",
            "start": earnings_date,
            "symbol": symbol,
            "eps_estimate": eps_estimate
        })

    return events

#########################################################
# ✅ 5) Streamlit UI
#########################################################
st.title("📊 글로벌 & 한국 주식 캘린더/공시 + 뉴스 RAG")

tab1, tab2 = st.tabs(["🌎 해외 주식", "🇰🇷 한국 주식"])

#########################
# 🌎 해외 주식 탭
#########################
with tab1:
    st.subheader("🌎 해외 주식 어닝 캘린더 + 뉴스 RAG")

    all_symbols = load_clean_us_symbols()
    #st.write(f"✅ 전체 미국 주식 {len(all_symbols)}개 로드 완료!")

    user_symbols = st.multiselect(
        "🔍 관심 있는 해외 종목 선택",
        options=all_symbols,
        default=["AAPL", "MSFT", "TSLA"]
    )

    if user_symbols:
        company_names = fetch_company_names(user_symbols)
        st.subheader("✅ 선택된 종목 리스트")
        for sym in user_symbols:
            st.write(f"- {sym} ({company_names[sym]})")

    if st.button("📆 해외 캘린더 표시"):
        earnings_events = get_earnings_calendar(user_symbols)
        st.session_state["calendar_events"] = earnings_events
        if not earnings_events:
            st.warning("⚠️ 선택한 종목에 어닝 일정 데이터가 없습니다. (yfinance 제한)")

    calendar_events = st.session_state.get("calendar_events", [])

    if calendar_events:
        st.subheader("🗓 어닝 일정 캘린더")
        calendar_options = {"initialView": "dayGridMonth", "selectable": True}
        calendar_return = calendar(events=calendar_events, options=calendar_options)

        if (
            calendar_return
            and "eventClick" in calendar_return
            and "event" in calendar_return["eventClick"]
        ):
            selected_event = calendar_return["eventClick"]["event"]
            symbol = selected_event["extendedProps"]["symbol"]

            company_name = fetch_company_names([symbol]).get(symbol, "")
            st.info(f"✅ 선택한 일정: {symbol} ({company_name}) | {selected_event['start']}")

            keyword = f"{symbol} {company_name}"
            query = f"{symbol} ({company_name}) 최근 어닝콜 관련 투자 포인트 요약해줘"

            if (
                "last_selected_symbol" not in st.session_state
                or st.session_state["last_selected_symbol"] != symbol
            ):
                with st.spinner(f"📰 {symbol} 뉴스 크롤링 및 RAG 분석 중..."):
                    crawl_naver_view_titles(keyword, limit=10)
                    create_faiss_index(keyword)
                    summary = rag_query(keyword, query)
                st.session_state["last_selected_symbol"] = symbol
                st.session_state["last_summary"] = summary

            if "last_summary" in st.session_state:
                st.success(f"🤖 분석 결과:\n\n{st.session_state['last_summary']}")

#########################
# 🇰🇷 한국 주식 탭
#########################
with tab2:
    st.subheader("🇰🇷 한국 실적공시 캘린더 + 뉴스/공시 결합 RAG")

    # ✅ 한국 상장사 리스트 불러오기
    corp_df = get_corp_list()
    if corp_df.empty:
        st.warning("⚠️ 한국 상장사 데이터를 불러오지 못했습니다.")
    else:
        corp_names = corp_df["corp_name"].tolist()
        selected_corp = st.selectbox("🔍 검색할 한국 기업", corp_names)

        if selected_corp:
            corp_code = corp_df[corp_df["corp_name"] == selected_corp]["corp_code"].values[0]

            # ✅ ‘실적 관련 공시’ 캘린더 표시
            if st.button("📆 한국 실적공시 캘린더 표시"):
                disclosures = get_recent_disclosures(corp_code)

                if disclosures:
                    events = []
                    for d in disclosures:
                        # YYYYMMDD → YYYY-MM-DD 변환
                        dt_fmt = f"{d['rcept_dt'][:4]}-{d['rcept_dt'][4:6]}-{d['rcept_dt'][6:]}"
                        events.append({
                            "title": f"{selected_corp} | {d['report_nm']}",
                            "start": dt_fmt,
                            "corp_name": selected_corp,
                            "report_nm": d["report_nm"],
                            "rcept_no": d["rcept_no"]
                        })
                    st.session_state["kr_events"] = events
                else:
                    st.warning("⚠️ 최근 90일간 실적 관련 공시가 없습니다.")

    kr_events = st.session_state.get("kr_events", [])
    if kr_events:
        st.subheader("🗓 한국 실적공시 캘린더")
        cal_ret = calendar(events=kr_events, options={"initialView": "dayGridMonth"})

        # ✅ 캘린더 클릭 → 공시+뉴스 결합 RAG
        if cal_ret and "eventClick" in cal_ret and "event" in cal_ret["eventClick"]:
            evt = cal_ret["eventClick"]["event"]
            cname = evt["extendedProps"]["corp_name"]
            rname = evt["extendedProps"]["report_nm"]
            rno = evt["extendedProps"]["rcept_no"]

            st.info(f"✅ {cname} | {rname} 뉴스+공시 분석 실행중...")
            with st.spinner("공시+뉴스 결합 RAG 분석 중..."):
                summary = analyze_disclosure_with_rag(cname, rname, rno)
            st.success(summary)