"""
FastAPI Swagger UI의 OpenAPI 스펙을 YAML 파일로 내보내는 스크립트
"""
import yaml
from app.main import app

def generate_openapi_yaml(output_path: str = "openapi.yaml"):
    """
    FastAPI 애플리케이션의 OpenAPI 스키마를 YAML 파일로 저장
    (Swagger UI에 표시되는 것과 동일한 내용)
    
    Args:
        output_path: 출력할 YAML 파일 경로 (기본값: openapi.yaml)
    """
    # OpenAPI 스키마를 딕셔너리로 가져오기 (Swagger UI에서 보는 것과 동일)
    openapi_schema = app.openapi()
    
    # YAML 형식으로 변환하여 파일에 저장
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(openapi_schema, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    
    print(f"✅ OpenAPI YAML 파일이 생성되었습니다: {output_path}")
    print(f"📄 이 파일은 Swagger UI (/docs)에서 보는 것과 동일한 API 스펙입니다.")

if __name__ == "__main__":
    generate_openapi_yaml()

