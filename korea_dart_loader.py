import requests, zipfile, io, xml.etree.ElementTree as ET, os, datetime
from bs4 import BeautifulSoup
from crawler import crawl_naver_view_titles
from rag_index import create_faiss_index_from_docs
from rag_search import rag_query_from_docs
from openai import OpenAI
import streamlit as st

clova_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_BASE_URL", "")
)

DART_API_KEY = st.secrets.get("DART_API_KEY", "")

# ================================
# ✅ 1. document.xml API 호출
# ================================
def fetch_disclosure_xml(rcept_no: str) -> str:
    """
    ✅ DART document.xml API → 공시 XML 원문 반환
    """
    url = "https://opendart.fss.or.kr/api/document.xml"
    params = {
        "crtfc_key": DART_API_KEY,
        "rcept_no": rcept_no
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        if res.status_code != 200:
            print(f"⚠️ 공시 XML 요청 실패: {res.status_code}")
            return ""
        return res.text
    except Exception as e:
        print(f"⚠️ 공시 XML 요청 오류: {e}")
        return ""

def fetch_disclosure_with_tables(rcept_no: str) -> str:
    """
    ✅ XML 기반 공시 본문 + 표 데이터 추출
    """
    xml_content = fetch_disclosure_xml(rcept_no)
    if not xml_content:
        return ""

    soup = BeautifulSoup(xml_content, "lxml")

    # 본문 텍스트
    paragraphs = [
        tag.get_text(strip=True)
        for tag in soup.find_all(["p", "div", "span", "tt"])
        if tag.get_text(strip=True)
    ]

    # 표 데이터 추출
    table_texts = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if any(cols):
                rows.append("\t".join(cols))
        if rows:
            table_texts.append("\n".join(rows))

    merged_tables = "\n\n".join(table_texts)

    return f"""
    ### 본문 ###
    {'\n'.join(paragraphs[:200])}  

    ### 표 데이터 ###
    {merged_tables}
    """

# ================================
# ✅ 2. 긴 텍스트 chunk 요약
# ================================
def chunk_text(text: str, max_chars=3500):
    """
    긴 텍스트를 max_chars 단위로 분리
    """
    chunks = []
    while len(text) > max_chars:
        split_idx = text[:max_chars].rfind(".")  # 문장 단위로 자르기
        if split_idx == -1:
            split_idx = max_chars
        chunks.append(text[:split_idx])
        text = text[split_idx:]
    if text:
        chunks.append(text)
    return chunks

def summarize_chunk_with_clova(chunk: str) -> str:
   
    prompt = f"""
    아래 공시 일부를 **핵심 요약**으로 정리해줘.
    - 표 안 숫자는 유지
    - 보고서 재무 관련 핵심 데이터를 요약하면서 비교 강조

    내용:
    {chunk}
    """
    try:
        res = clova_client.chat.completions.create(
            model="HCX-005",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ OpenAI chunk 요약 실패: {e}")
        return chunk[:1000]  # 실패하면 일부만 반환

def safe_summarize_large_text(full_text: str) -> str:
    """
    ✅ 너무 긴 공시는 chunk 요약 후 최종 압축
    """
    if len(full_text) < 4000:
        return full_text  # 짧으면 그대로 사용

    chunks = chunk_text(full_text, max_chars=3500)
    partial_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"📝 Chunk {i}/{len(chunks)} 요약 중...")
        summary = summarize_chunk_with_clova(chunk)
        partial_summaries.append(summary)

    # ✅ partial 요약을 다시 압축
    final_prompt = f"""
    아래 여러 개의 요약을 **한 문서로 통합 요약**해줘.
    - 재무 숫자는 유지 매출, 영업이익, 순이익 변화율 강조
    - 초보 투자자가 이해할 수 있는 3~4줄 요약
    - 핵심 요약만 남기고 불필요한 문장은 제거

    {'\n\n'.join(partial_summaries)}
    """
    try:
        res = clova_client.chat.completions.create(
            model="HCX-005",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ 최종 통합 요약 실패: {e}")
        return "\n\n".join(partial_summaries)

# ================================
# ✅ 3. 공시 + 뉴스 → RAG docs
# ================================
def analyze_disclosure_with_rag(corp_name, report_nm, rcept_no):
    """
    ✅ 공시 본문 + 표 → chunk 요약 → 뉴스 결합 → RAG 시나리오
    """
    # 1) 공시 본문 + 표 데이터
    disclosure_raw = fetch_disclosure_with_tables(rcept_no)
    if not disclosure_raw:
        return "⚠️ 공시 원문을 불러오지 못했습니다."

    # ✅ 길면 chunk 요약 적용
    disclosure_text = safe_summarize_large_text(disclosure_raw)

    # 2) 관련 뉴스 크롤링
    keyword = f"{corp_name} {report_nm}"
    news_results = crawl_naver_view_titles(keyword, limit=5) or []
    news_docs = [n.get("preview") for n in news_results if n.get("preview")]

    # 3) docs 결합
    docs = []
    docs.append(f"[공시 요약] {corp_name} - {report_nm}\n{disclosure_text}")
    docs.extend([f"[뉴스] {nd}" for nd in news_docs])

    if not docs:
        return "⚠️ 공시/뉴스 데이터가 없습니다."

    # ✅ 벡터 인덱스 생성
    save_path = f"embeddings/{corp_name}_mix.faiss"
    create_faiss_index_from_docs(docs, save_path=save_path)

    # ✅ 프롬프트
    query = f"""
    {corp_name}의 '{report_nm}' 공시에서 **표 안의 재무 숫자**를 중심으로
    - 매출, 영업이익, 순이익, 전년동기/전분기 대비 변화율
    - 시장 컨센서스 대비 평가
    - 주가 영향 시나리오 (호재/중립/악재)
    - 관련 섹터 대체 전략
    요약해줘.
    """
    return rag_query_from_docs(query, docs, save_path)

# ================================
# ✅ 4. 상장사 리스트 & 최근 공시 필터
# ================================
def get_corp_list(save_path="data/corp_list.csv"):
    """
    ✅ DART 전체 상장사 리스트 (corp_code 매핑)
    """
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={DART_API_KEY}"
    res = requests.get(url)
    if res.status_code != 200:
        raise Exception("DART API 연결 실패")

    with zipfile.ZipFile(io.BytesIO(res.content)) as z:
        xml_file = z.open(z.namelist()[0]).read()

    root = ET.fromstring(xml_file)
    rows = []
    for row in root.findall("list"):
        rows.append([
            row.find("corp_code").text,
            row.find("corp_name").text,
            row.find("stock_code").text
        ])

    import pandas as pd
    df = pd.DataFrame(rows, columns=["corp_code", "corp_name", "stock_code"])
    os.makedirs("data", exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df

def get_recent_disclosures(corp_code=None, start_date=None, end_date=None, page_count=50):
    """
    ✅ 최근 ‘실적 관련’ 공시만 필터
    """
    if not start_date:
        start_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime("%Y%m%d")
    if not end_date:
        end_date = datetime.date.today().strftime("%Y%m%d")

    url = f"https://opendart.fss.or.kr/api/list.json?crtfc_key={DART_API_KEY}&bgn_de={start_date}&end_de={end_date}&page_count={page_count}"
    if corp_code:
        url += f"&corp_code={corp_code}"

    res = requests.get(url).json()
    if res.get("status") != "000":
        print(f"⚠️ DART API 오류: {res.get('message')}")
        return []

    keywords = ["분기보고서", "반기보고서", "사업보고서", "잠정실적"]
    filtered = []
    for d in res.get("list", []):
        if any(k in d["report_nm"] for k in keywords):
            dt = d["rcept_dt"]
            filtered.append({
                "corp_name": d["corp_name"],
                "report_nm": d["report_nm"],
                "rcept_dt": dt,
                "rcept_no": d["rcept_no"],
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={d['rcept_no']}"
            })
    return filtered