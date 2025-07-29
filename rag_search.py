import faiss
import numpy as np
import json
import os
import time
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ✅ Clova API (OpenAI 호환)
client = OpenAI(
    api_key="--",
    base_url="--"
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ API 호출 캐싱 (같은 요청 다시 안 함)
API_CACHE = {}

# ✅ API 호출 쿨타임 관리
LAST_CALL_TIME = 0
COOLDOWN_SECONDS = 10  # 10초 쿨타임

# =====================================================
# ✅ 공통: Clova API 안전 호출 (429 → 재시도)
# =====================================================
def safe_clova_call(prompt, retry=3):
    global LAST_CALL_TIME

    # ✅ 쿨타임 적용
    now = time.time()
    if now - LAST_CALL_TIME < COOLDOWN_SECONDS:
        wait_time = int(COOLDOWN_SECONDS - (now - LAST_CALL_TIME))
        print(f"⏳ 요청이 너무 많음 → {wait_time}초 대기")
        time.sleep(wait_time)

    for attempt in range(retry):
        try:
            res = client.chat.completions.create(
                model="HCX-005",
                messages=[{"role": "user", "content": prompt}]
            )
            LAST_CALL_TIME = time.time()
            return res.choices[0].message.content
        except Exception as e:
            # 429 Too Many Requests → 재시도
            if "429" in str(e):
                wait = 5 * (attempt + 1)
                print(f"⚠️ Clova 429 Too Many Requests → {wait}초 후 재시도 ({attempt+1}/{retry})")
                time.sleep(wait)
                continue
            else:
                return f"⚠️ Clova API 호출 실패: {e}"

    return "⚠️ Clova API 재시도 실패 (요청 한도 초과)"

# =====================================================
# ✅ 해외 뉴스 RAG
# =====================================================
def rag_query(keyword, query):
    """
    ✅ 해외 뉴스 JSON 기반 RAG
    - 크롤링한 data/{keyword}.json 읽어서 벡터검색 후 시나리오
    """

    # ✅ 캐시 키 확인 → 같은 질문은 다시 안 함
    cache_key = f"news::{keyword}::{query}"
    if cache_key in API_CACHE:
        print(f"✅ 캐싱된 요약 반환: {cache_key}")
        return API_CACHE[cache_key]

    index_path = f"embeddings/{keyword}_index.faiss"
    json_path = f"data/{keyword}.json"

    if not os.path.exists(index_path) or not os.path.exists(json_path):
        return "⚠️ 관련 뉴스 데이터가 없습니다."

    # ✅ 벡터 인덱스 로드
    idx = faiss.read_index(index_path)

    # ✅ 뉴스 JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        news_data = json.load(f)

    docs = [
        item.get("preview", "").strip()
        for item in news_data
        if item.get("preview") and len(item["preview"].strip()) > 50
    ]
    if not docs:
        return "⚠️ 유효한 뉴스 문서가 없습니다."

    # ✅ 쿼리 임베딩 후 검색
    q_emb = embed_model.encode([query])
    _, I = idx.search(np.array(q_emb), k=1)

    # ✅ 상위 문서
    context = docs[I[0][0]]
    if len(context) > 1000:
        context = context[:1000]

    prompt =f"""
너는 **해외 기업 실적 발표(어닝콜)를 분석해주는 투자 분석 전문가 AI**야.  
분석 대상은 **투자 초보자**이기 때문에, 친근하면서도 정확하고 구체적으로 설명해줘야 해.

---

 **분석 시 반드시 지켜야 할 규칙**

1. 실적 수치는 절대 바꾸지 말고 뉴스에 나온 그대로 유지해.
   - 매출, EPS(주당순이익), 성장률 등
   - “예상치 대비 얼마나 차이 났는지” 명확히 적시
2. 전문 용어는 초보자가 이해할 수 있도록 풀어서 설명해줘.
   - EPS → 주당순이익
   - YoY → 전년 동기 대비
   - Guidance → 회사가 전망한 향후 실적 예상
3. “그래서 주가에 긍정적인지, 부정적인지” 명확히 판단해줘.
4. CEO 발언, 주주환원(자사주 매입 등), 성장전략 등도 언급되었으면 꼭 포함시켜.
5. 시나리오는 최대한 구체적으로, 주가가 어떤 조건에서 어떻게 움직일 수 있을지 예측해줘.

---

 **어닝 뉴스 본문**  
{context}

---

✍️ **출력 형식**

1 **핵심 요약 (2~3줄)**  
- EPS, 매출이 시장 예상보다 얼마나 좋았는지/나빴는지  
- CEO 발언이나 주요 이슈가 있었는지 간략히 요약  
- 초보 투자자가 직관적으로 이해할 수 있는 문장으로 표현  


2 **시장 컨센서스와 비교 분석**  
- 시장 컨센스가 예상보다 좋았고, 어떤 건 아쉬웠는지  
- 투자자들이 실망했을 만한 포인트는 무엇인지  
- 기대보다 좋았음에도 주가가 하락했다면, 그 이유도 설명 (ex. 가이던스 하향)

3 **주가 영향 시나리오 (핵심)**  
각 시나리오는 **숫자 + 조건 + 시장 반응 예상**을 함께 제시해줘

-  **호재 시나리오 (상승 가능)**  
→ EPS + 매출 모두 서프라이즈, 향후 성장 기대감, 자사주 매입 등  
→ 주가 +5~10% 상승 가능 조건 제시

-  **중립 시나리오 (변동성 낮음)**  
→ 예상치와 비슷, 특별한 이슈 없음  
→ 주가 +1~2% 또는 보합

- **악재 시나리오 (하락 가능성)**  
→ 실적 미달, 전망 하향, 매크로 불확실성 등  
→ 주가 -3~10% 하락 위험 구체적 근거 제시

→ 시나리오 예측에 신뢰성을 더하기 위해 **최근 비슷한 기업의 사례**나 시장 반응도 참조 가능

4 **초보 투자자 행동 가이드 (1줄)**  
→ “지금은 관망이 좋아요”, “분할 매수 고려 가능”, “리스크 크므로 신중히 접근” 등  
→ 반드시 위 시나리오 기반으로 결론 도출

---

 **절대 숫자 임의로 만들지 마! 뉴스 기반으로만 판단해!**  
 **어려운 용어는 설명하거나 바꿔서 초보자도 이해 가능하게!**  
 **시나리오가 핵심이다. 각 조건과 반응을 구체적으로!**
"""
    # ✅ 안전 API 호출
    answer = safe_clova_call(prompt)

    # ✅ 캐싱 저장
    API_CACHE[cache_key] = answer
    return answer

# =====================================================
# ✅ 한국 공시 RAG
# =====================================================
def rag_query_from_docs(query, docs, index_path):
    """
    ✅ 한국 공시 → OpenAI GPT-4o-mini로만 처리
    """
    if not os.path.exists(index_path):
        return "⚠️ RAG 인덱스가 없습니다."

    idx = faiss.read_index(index_path)
    q_emb = embed_model.encode([query])
    _, I = idx.search(np.array(q_emb), k=2)

    context = "\n\n".join([docs[i] for i in I[0]])

    prompt = f"""
너는 **초보 투자자들을 위한 재무 분석 AI**이자,  
동시에 **현실적인 투자 시나리오를 제시하는 투자 전략 어드바이저**야.

 분석 목적:
- 초보 투자자가 이해할 수 있는 핵심 요약
- 공시 수치 기반 변화 요약 + 시장 반응 예상
- 구체적인 투자 판단 시나리오 제공 (주가 반등 or 리스크 경고)

---

 **공시 원문 (표 포함)**  
{context}

---

✍️ **출력 형식**

---

1 **공시 원본 표 기반 요약**  
→ 숫자는 절대 바꾸지 말고 기반하여 표로 핵심 요약을 제공해줘

---

2 **한눈에 보는 핵심 요약 (2~3줄)**  
- 매출/영업이익/순이익의 전년 대비, 전분기 대비 변화  
- 간단한 문장으로 좋은 소식인지 나쁜 소식인지 판단 근거 포함

---

3 **시장 컨센서스와 비교**  
- 증권가 예상치 대비 얼마나 차이 나는지 (예상치 없으면 '미제공' 명시)
- 숫자 차이와 그 의미를 설명해줘 (ex. 매출은 높았지만 이익률은 낮음 등)

---

4 **고도화된 투자 시나리오 (핵심)**  
각 시나리오에서 “왜” 그런 판단이 나오는지를 뉴스와 데이터 기반으로 충분히 뒷받침해줘.

-  **호재 시나리오 (상승 가능)**  
→ 어떤 조건(신사업, 수주 증가, 이익률 반등 등)에서 주가가 반등할 수 있는지  
→ 크롤링한 뉴스/업종 트렌드 참고해서 구체적이고 현실적으로

-  **중립 시나리오 (관망)**  
→ 실적은 나쁘지 않지만 아직 모멘텀 부족한 이유  
→ 외부 리스크 요인(환율, 금리, 글로벌 변수 등)도 고려

- **악재 시나리오 (하락 위험)**  
→ 실적 부진 외에도 주가 하락 압력을 주는 구조적 문제  
→ 예: 고정비 부담, 실적 휘발성, 경쟁 심화, 수요 감소 등

각 시나리오에서 "주가가 실제로 어떻게 반응할 가능성이 있는지"도 예시처럼 간단히 숫자로 표현해줘 (ex. “5% 반등 여지”)

---

5 **초보 투자자 행동 가이드 (1줄)**  
- 지금은 관망인지, 분할 매수인지, 손절 타이밍인지  
- 단순 감정이 아니라 “시나리오 분석 기반”으로 결정해야 함

→ 예시:  
- “저가 매수 기회일 수 있어요. 거래량 증가 여부를 지켜보세요.”  
- “실적 하락과 함께 모멘텀 부족. 지금은 관망이 좋습니다.”  
- “호재는 있지만 시장 전반이 불안정하므로 신중한 접근 필요.”

---

 **절대 표 속 숫자 바꾸지 마!**  
 **표→요약→시나리오→행동 순서로 논리 흐름을 유지해줘**  
 **초보자도 이해 가능한 쉬운 언어와 간결한 설명만!**
"""

    try:
        # ✅ 한국 공시는 바로 OpenAI GPT 사용
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        ) 
        return res.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI 호출 실패: {e}"