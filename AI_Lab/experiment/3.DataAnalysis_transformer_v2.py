"""
이 코드는 모든 *_jobs.json 파일에서 스킬을 수집하고, transformers 기반 임베딩 및 UMAP 시각화를 제공합니다.
- 클릭시 클러스터 반경 내 스킬들을 팝업으로 보여주고, 검색으로 특정 스킬 위치를 하이라이트합니다.

아래의 이슈/의심 포인트 및 원인 진단도 프린트와 주석으로 추가되어 있습니다.

- 증상: 모든 스킬이 시각화(plot/UMAP 등)에 나타나지 않거나 일부만 빠질 경우?
- 원인1: 입력 skills_list에 중복/공란/None 등이 들어갈 경우
- 원인2: encode 과정에서 내부적으로 에러가 발생했으나, try-except로 잡히지 않음
- 원인3: plotly (혹은 pandas DataFrame) 생성시 row 누락
- 원인4: transformers 모델 또는 tokenizer 이슈로 인코딩 누락

입력 정제, 파일로부터 데이터 수집, 진단 프린트 및 임베딩/시각화까지 자동 수행됩니다.
"""

import os
import glob
import json
import numpy as np

def load_all_skills_from_jobs(data_dir=None):
    """
    모든 *_jobs.json 파일을 읽고, 내부의 모든 고유한 스킬을 리스트로 반환합니다.
    """
    # 데이터 폴더 결정
    if data_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, 'data')

    print(f"[INFO] 데이터 디렉토리: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"[ERROR] 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return []

    job_files = glob.glob(os.path.join(data_dir, "*_jobs.json"))
    if not job_files:
        print("[ERROR] *_jobs.json 파일을 찾을 수 없습니다.")
        return []

    unique_skills = set()
    for fp in job_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
                for job in jobs:
                    skills = []
                    if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                        skills = job['skill_set_info']['skill_set']
                        if not isinstance(skills, list):
                            continue
                        for s in skills:
                            if s is not None and str(s).strip() and str(s).lower() != "none":
                                unique_skills.add(str(s).strip())
        except Exception as e:
            print(f"[WARN] 파일 읽기 에러: {fp}: {e}")

    print(f"[INFO] 전체 공고에서 추출한 고유 스킬 수: {len(unique_skills)}")
    return list(unique_skills)


def vectorize_skills_with_transformers(skills_list, model_name=None):
    """
    입력받은 스킬 목록(skills_list)을 transformers 기반 임베딩 벡터로 반환

    ⚠️ 만약 임베딩 결과에서 일부 스킬이 빠지거나 shape이 맞지 않으면
    - skills_list 내 개별 스킬의 중복/None/공백/이상값 유무를 먼저 확인하세요.
    - encode 처리 과정에서 에러 발생해도 일부만 반환될 수 있음.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[ERROR] sentence-transformers 패키지가 필요합니다.")
        print("설치: pip install sentence-transformers")
        return None

    # 1. 입력 검증 및 정제
    cleaned_skills = []
    for i, s in enumerate(skills_list):
        if not s or str(s).strip() == "" or str(s).lower() == "none":
            print(f"[WARN] 스킬 [{i}] 값이 비어있음/None: {repr(s)} --> 제거")
            continue
        cleaned_skills.append(str(s).strip())
    if len(set(cleaned_skills)) < len(cleaned_skills):
        print(f"[INFO] 중복 스킬 {len(cleaned_skills)-len(set(cleaned_skills))}개 발견 -> 중복 제거")
        cleaned_skills = list(dict.fromkeys(cleaned_skills))  # 순서 유지 중복제거

    print(f"[INFO] 최종 임베딩 입력 스킬 수: {len(cleaned_skills)}/{len(skills_list)}")

    # 2. 모델 후보군 (2번 코드와 동일)
    if model_name is None:
        model_options = [
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'jhgan/ko-sbert-multitask',
            'sentence-transformers/all-MiniLM-L6-v2',
        ]
    else:
        model_options = [model_name]

    model = None
    for mopt in model_options:
        try:
            model = SentenceTransformer(mopt)
            print(f"[INFO] 사용 모델 로드 성공: {mopt}")
            break
        except Exception as e:
            print(f"[WARN] 모델 로드 실패: {mopt}: {e}")
            continue

    if model is None:
        print("[ERROR] transformers 기반 임베딩 모델을 모두 로드하지 못했습니다.")
        return None

    # 3. 임베딩 생성
    try:
        embeddings = model.encode(
            cleaned_skills,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 오류: {e}")
        return None

    if len(embeddings) != len(cleaned_skills):
        print(f"[ERROR] 임베딩 결과 개수 불일치! ({len(embeddings)}개 != {len(cleaned_skills)}개)")
        print("입력 및 예외 케이스를 확인하세요.")

    print(f"[SUCCESS] {len(embeddings)}개 스킬 임베딩 완료. shape={embeddings.shape}")
    return embeddings, cleaned_skills  # cleaned_skills도 반환 (아래 시각화에서 동기화 중요)


def find_centroids_and_labels(embeddings, n_clusters=10, random_state=42):
    """
    클러스터링을 통해 임베딩의 대표 중심점 기술 인덱스를 반환합니다.
    (여기서는 KMeans를 사용합니다.)
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("[ERROR] scikit-learn 패키지가 필요합니다. pip install scikit-learn")
        return None, None, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    # 각 클러스터 중심과 가장 가까운 점(기술) 인덱스 찾기 (대표 기술)
    from scipy.spatial import distance
    centroid_indices = []
    for i, center in enumerate(centers):
        dists = distance.cdist([center], embeddings[labels == i])
        if dists.shape[1] == 0:
            # 해당 클러스터(군집)에 아무 기술이 없는 특이 케이스!
            centroid_indices.append(None)
            continue
        min_idx_in_cluster = np.argmin(dists)
        cluster_indices = np.where(labels == i)[0]
        centroid_indices.append(cluster_indices[min_idx_in_cluster])
    return centroid_indices, labels, centers

def plot_umap_interactive_with_cluster_click_and_search(
    embeddings, skills_list, n_clusters=10, title="UMAP 클러스터 시각화 (검색/클릭 기능 포함)", radius=0.25
):
    """
    UMAP + KMeans 클러스터링 결과를 plotly로 시각화합니다.
    - 각 클러스터 대표점에 '클릭'하면 해당 중심점 반경 내 skill들 리스트를 popup(hover 등)으로 출력
    - 텍스트 검색창에서 스킬명 입력 시, 해당 스킬 자동 하이라이트
    radius: 군집 중심에서의 유사 기술 탐색 반경 (UMAP 좌표 거리 기준)
    """
    try:
        import umap
    except ImportError:
        print("[ERROR] umap-learn 패키지가 필요합니다. pip install umap-learn")
        return
    try:
        import plotly.graph_objects as go
        import pandas as pd
        from plotly.subplots import make_subplots
    except ImportError:
        print("[ERROR] plotly, pandas 패키지가 필요합니다. pip install plotly pandas")
        return

    if len(embeddings) != len(skills_list):
        print(f"[ERROR] 임베딩과 스킬명 개수 불일치: emb={len(embeddings)}, skills={len(skills_list)}")
        return

    centroid_indices, labels, centers = find_centroids_and_labels(embeddings, n_clusters=n_clusters)
    if centroid_indices is None:
        print("[ERROR] 대표 중심 기술 계산 실패")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    centers_2d = reducer.transform(centers)

    # 모든 점, 대표점, 클러스터 정보 데이터프레임 준비
    df_all = pd.DataFrame({
        "x": embeddings_2d[:,0],
        "y": embeddings_2d[:,1],
        "skill": skills_list,
        "cluster": [f"Cluster {i+1}" for i in labels]
    })
    # 대표점 정보 (centroids)
    df_centroid = pd.DataFrame({
        "x": centers_2d[:,0],
        "y": centers_2d[:,1],
        "skill": [skills_list[idx] if idx is not None else "" for idx in centroid_indices],
        "cluster_num": [i for i in range(n_clusters)],
        "cluster": [f"Cluster {i+1}" for i in range(n_clusters)],
        "idx": centroid_indices
    })

    # 각 대표점 반경 내에 있는 기술 목록 생성
    from scipy.spatial import distance

    neighbor_skills = []
    for k in range(n_clusters):
        cx, cy = centers_2d[k]
        mask = np.where(labels == k)[0]
        group_pts = embeddings_2d[mask]
        dists = np.linalg.norm(group_pts - np.array([cx,cy]), axis=1)
        # 반경 내 또는 상위 n개만도 가능
        within = np.where(dists < radius)[0]
        if len(within) == 0:
            topn = np.argsort(dists)[:8]
            indices = mask[topn]
        else:
            indices = mask[within]
        skills_in_radius = [skills_list[i] for i in indices]
        neighbor_skills.append(skills_in_radius)
    df_centroid["neighbor_skills"] = neighbor_skills

    # plotly 그래프 생성
    fig = go.Figure()

    # 1. 전체 스킬 산점도 (흐릿하게)
    for k in range(n_clusters):
        sub_df = df_all[df_all["cluster"]==f"Cluster {k+1}"]
        fig.add_trace(go.Scatter(
            x=sub_df["x"],
            y=sub_df["y"],
            mode="markers",
            name=f"Cluster {k+1} (skills)",
            marker=dict(size=9, opacity=0.27),
            hoverinfo="text",
            text=sub_df["skill"],
            customdata=sub_df["skill"],
            visible=True,
        ))

    # 2. 대표점(centroid) star marker + 텍스트
    fig.add_trace(go.Scatter(
        x=df_centroid["x"], y=df_centroid["y"],
        mode="markers+text",
        name="대표 skill (centroid)",
        marker=dict(size=28, symbol="star", color="red", opacity=0.86, line=dict(width=3, color="black")),
        text=df_centroid["skill"],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>클러스터 주요기술:<br>%{customdata}',
        customdata=[", ".join(skills) for skills in df_centroid["neighbor_skills"]],
        showlegend=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="UMAP 차원 1",
        yaxis_title="UMAP 차원 2",
        width=950, height=730,
        hovermode="closest"
    )

    # 3. 클릭시 중심점 반경 내 기술 show - plotly events 이용(설치) ⇒ 대신 Dash app으로 wrapping!
    # 4. 검색 박스 추가 (Dash Input): skill명을 입력하면 해당 포인트를 highlight

    try:
        from dash import Dash, dcc, html, Input, Output, State
        import dash
        # callback_context는 함수 내부에서 사용
    except ImportError:
        print("[ERROR] dash 패키지가 필요합니다. pip install dash")
        fig.show()
        print("[WARN] dash 미설치시 상호작용/검색 기능 없이 plotly만 지원")
        return

    app = Dash(__name__)
    # Layout
    app.layout = html.Div([
        html.Div([
            dcc.Input(id="skill_search", type="text", placeholder="스킬명으로 검색...", style={"width":"240px", "marginRight":"10px"}),
            html.Button("검색", id="search_btn"),
        ], style={"marginBottom":"10px"}),
        dcc.Graph(id="umap_plot", figure=fig, style={"height":"700px"}),
        html.Div(id='clicked_info', style={'marginTop': '18px', 'whiteSpace':'pre-wrap', "fontSize":"17px"}),
    ])

    @app.callback(
        Output('umap_plot', 'figure'),
        Output('clicked_info','children'),
        Input('umap_plot', 'clickData'),
        Input('search_btn', 'n_clicks'),
        State('skill_search', 'value'),
        prevent_initial_call=True
    )
    def highlight_and_popup(clickData, n_clicks, search_val):
        # callback_context 가져오기
        try:
            ctx = dash.callback_context
        except:
            # 최신 Dash 버전에서는 다른 방식으로 접근
            try:
                from dash._callback_context import callback_context
                ctx = callback_context
            except:
                # callback_context를 사용할 수 없는 경우
                ctx = None
        
        fig_new = fig.to_dict()
        info_text = ""
        
        # callback_context에서 triggered 정보 추출
        triggered_id = None
        if ctx:
            try:
                if hasattr(ctx, 'triggered') and ctx.triggered:
                    # triggered는 리스트 형태: [{'prop_id': 'umap_plot.clickData', 'value': ...}]
                    prop_id = ctx.triggered[0].get('prop_id', '') if ctx.triggered else ''
                    triggered_id = prop_id.split('.')[0] if prop_id else None
            except:
                # callback_context 사용 실패 시, 입력 파라미터로 판단
                if clickData:
                    triggered_id = 'umap_plot'
                elif n_clicks and n_clicks > 0:
                    triggered_id = 'search_btn'
        else:
            # callback_context가 없는 경우, 입력 파라미터로 판단
            if clickData:
                triggered_id = 'umap_plot'
            elif n_clicks and n_clicks > 0:
                triggered_id = 'search_btn'
        
        if triggered_id == 'umap_plot' and clickData and "points" in clickData:
            # 클릭 위치가 centroid에 가까우면 info 출력: neighbor_skills
            px = clickData["points"][0]["x"]
            py = clickData["points"][0]["y"]
            min_dist = float('inf')
            min_idx = None
            for i, (cx, cy) in enumerate(zip(df_centroid["x"], df_centroid["y"])):
                d = np.linalg.norm(np.array([px,py])-np.array([cx,cy]))
                if d < min_dist:
                    min_dist = d
                    min_idx = i
            if min_idx is not None and min_dist < radius*1.7:
                skill_name = df_centroid.iloc[min_idx]['skill']
                neighbor_list = ", ".join(df_centroid.iloc[min_idx]["neighbor_skills"])
                info_text = f"■ {skill_name} (Cluster {min_idx+1} 대표)\n반경 내 기술: {neighbor_list}"
        elif triggered_id == "search_btn" and search_val:
            # 검색 결과: 해당 스킬 위치 하이라이트 + 센터링
            search_val = str(search_val).strip()
            sel_idx = df_all[df_all["skill"].str.lower() == search_val.lower()].index
            if len(sel_idx)==0:
                info_text = f"'{search_val}' 관련 스킬 없음."
            else:
                sel_idx = sel_idx[0]
                sel_row = df_all.iloc[sel_idx]
                info_text = f"▶ 검색: {sel_row['skill']} ({sel_row['cluster']})\n위치=({sel_row['x']:.3f}, {sel_row['y']:.3f})"
                # 새 Trace 추가 (강조 점)
                sel_trace = go.Scatter(
                    x=[sel_row["x"]], y=[sel_row["y"]],
                    mode="markers+text",
                    marker=dict(size=32, color="gold", symbol="diamond-open"),
                    text=[sel_row["skill"]],
                    textposition="bottom center",
                    showlegend=False,
                    name="검색 스킬"
                )
                # 기존 figure에 강조점 (마지막 trace로 add)
                fig_new["data"].append(sel_trace.to_plotly_json())
        else:
            info_text = ""
        
        # HTML 형식으로 변환 (문자열이면 그대로 반환, Dash는 자동으로 처리)
        return fig_new, info_text

    print("[INFO] Dash web app 실행: http://127.0.0.1:8050 (검색 및 클릭 팝업 지원)")
    app.run_server(debug=False)


if __name__ == "__main__":
    print("=== [STEP 1] 공고 데이터에서 전체 스킬 추출 ===")
    all_skills = load_all_skills_from_jobs()  # data 폴더에서 전체 스킬 추출
    print(f"스킬 예시 (상위 10개): {all_skills[:10]}")

    if not all_skills or len(all_skills) < 3:
        print("[ERROR] 충분한 스킬 데이터가 없습니다.")
    else:
        print("\n=== [STEP 2] 임베딩 및 UMAP 클러스터링 시각화(검색/클릭 지원) ===")
        result = vectorize_skills_with_transformers(all_skills)
        if result is not None:
            vectors, cleaned_skills = result
            print("임베딩 shape:", vectors.shape)
            print("cleaned_skills(상위 20개):", cleaned_skills[:20])
            plot_umap_interactive_with_cluster_click_and_search(
                vectors, cleaned_skills, n_clusters=10, radius=0.23
            )
