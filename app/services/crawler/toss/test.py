import networkx as nx
import plotly.graph_objects as go

# 유사도 임계값 이상인 공고들을 연결
G = nx.Graph()
for i, job_a in enumerate(jobs):
    for j, job_b in enumerate(jobs[i+1:]):
        similarity = cosine_similarity(job_a, job_b)
        if similarity > 0.7:  # 임계값
            G.add_edge(job_a['company'], job_b['company'], 
                      weight=similarity)