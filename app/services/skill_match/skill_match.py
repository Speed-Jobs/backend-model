import os
import pickle
import numpy as np

# 현재 파일 기준 상대 경로로 model 파일 위치 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../../core/data_model/skill_association_model.pkl')

def load_skill_association_model(model_path):
    """
    저장된 Node2Vec 기반 스킬 연관성 모델을 불러옵니다.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model']

def get_similar_skills(model, keywords, top_n=10):
    """
    키워드(스킬 목록이나 하나의 스킬)를 입력하면 유사한 스킬셋을 반환합니다.
    Args:
        model: loaded gensim/word2vec model from pickle data
        keywords: str or list of str (스킬 키워드)
        top_n: 반환할 유사 스킬 갯수
    Returns:
        [(skill(str), score(float)), ...]
    """

    print("keywords: ", keywords)
    if isinstance(keywords, str):
        keywords = [keywords]
    # 존재하는 것만으로 필터링
    valid_keywords = [k for k in keywords if k in model.wv]
    if not valid_keywords:
        print(f"입력 스킬이 모델에 존재하지 않습니다: {keywords}")
        return []
    # 단일 스킬이면 most_similar
    if len(valid_keywords) == 1:
        return model.wv.most_similar(valid_keywords[0], topn=top_n)
    else:
        # 여러 스킬일 경우 컨텍스트(평균 벡터)로 추천
        vectors = [model.wv[k] for k in valid_keywords]
        avg = np.mean(vectors, axis=0)
        sims = model.wv.similar_by_vector(avg, topn=top_n + len(valid_keywords))
        return [(skill, score) for skill, score in sims if skill not in valid_keywords][:top_n]

def print_similar_skills(model, keyword, top_n=10):
    """
    유사 스킬셋을 출력
    """
    print("=" * 60)
    if isinstance(keyword, list):
        k_disp = ", ".join(keyword)
    else:
        k_disp = str(keyword)
    print(f"입력한 키워드: {k_disp}")
    print(f"유사한 스킬 Top {top_n}:")
    print("=" * 60)
    similar = get_similar_skills(model, keyword, top_n=top_n)
    for skill, score in similar:
        print(f"  {skill}: {score:.4f}")



def main():
    print("=" * 60)
    print("Skill Association Similarity Test - (Pretrained Model)")
    print("=" * 60)
    # 모델 로드
    model = load_skill_association_model(MODEL_PATH)
    print("모델 로드 완료")

    while True:
        print("\n유사 스킬을 찾고 싶은 키워드를 입력하세요.")
        print("여러 스킬은 콤마(,)로 구분해서 입력할 수 있습니다. (종료하려면 'exit' 입력)")
        user_input = input("입력 키워드: ").strip()
        if user_input.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        if not user_input:
            print("입력이 비어 있습니다. 다시 입력하세요.")
            continue
        # 쉼표가 있으면 리스트로 처리
        if "," in user_input:
            keywords = [k.strip() for k in user_input.split(",") if k.strip()]
        else:
            keywords = [user_input]
        print_similar_skills(model, keywords, top_n=10)

    # 전체 보유 스킬 일부 리스트 출력
    print("\n" + "="*60)
    print("모델에 포함된 스킬(샘플):")
    all_skills = list(model.wv.index_to_key)
    print(f"총 {len(all_skills)}개 스킬")
    print("샘플 20개:", all_skills[:20])

if __name__ == "__main__":
    main()
