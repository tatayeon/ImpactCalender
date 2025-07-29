from bs4 import BeautifulSoup
import requests
import json
import os

def crawl_naver_view_titles(keyword, limit=10):
    base_url = "https://search.naver.com/search.naver?where=view&sm=tab_jum&query="
    extra_url = "&sm=tab_smr&sort=0&ssc=tab.news.all"
    search_url = base_url + keyword + extra_url

    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    items = soup.select(".sds-comps-vertical-layout.sds-comps-full-layout._sghYQmdqcpm83O1jqen")

    # sds-comps-vertical-layout.sds-comps-full-layout._sghYQmdqcpm83O1jqen

    results = []

    for rank_num, item in enumerate(items[:limit], 1):
        # 제목 + 미리보기 텍스트
        text = item.get_text(separator=" ", strip=True)

        # 링크 추출
        link_tag = item.select_one("a.api_txt_lines.total_tit")
        url = link_tag["href"] if link_tag else None
        title = link_tag.text.strip() if link_tag else "제목 없음"

        results.append({
            "rank": rank_num,
            "title": title,
            "preview": text,
            "url": url
        })

        print(f"{rank_num}. {title}")
        print(f"URL: {url}")
        print()

    # 저장
    if results:
        os.makedirs("data", exist_ok=True)
        with open(f"data/{keyword}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"✅ 저장 완료: data/{keyword}.json")
    else:
        print("❌ 저장할 결과가 없습니다.")
    return results   # ✅ 반드시 반환!

# if __name__ == "__main__":
#     keyword = input("검색할 키워드를 입력하세요: ")
#     crawl_naver_view_titles(keyword, limit=10)
