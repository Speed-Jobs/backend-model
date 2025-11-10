import json

# 카카오 채용공고 로드
with open('data/kakao_jobs.json', 'r', encoding='utf-8') as f:
    kakao = json.load(f)

# 우아한형제들 채용공고 로드
with open('data/woowahan_jobs.json', 'r', encoding='utf-8') as f:
    woowahan = json.load(f)

print(f"카카오 공고 수: {len(kakao)}개")
print(f"우아한형제들 공고 수: {len(woowahan)}개")
print(f"총 공고 수: {len(kakao) + len(woowahan)}개")

# 카카오 첫 공고 샘플
print("\n[카카오 첫 번째 공고 샘플]")
print(f"제목: {kakao[0]['title']}")
print(f"회사: {kakao[0]['company']}")
print(f"경력: {kakao[0]['experience']}")

# 우아한형제들 첫 공고 샘플
print("\n[우아한형제들 첫 번째 공고 샘플]")
print(f"제목: {woowahan[0]['title']}")
print(f"회사: {woowahan[0]['company']}")
print(f"경력: {woowahan[0]['experience']}")

