"""
S3 업로드 유틸리티 (크롤러 스크린샷용)
"""
import os
from typing import Optional
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


# .env 로드
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# S3 접속 정보 (환경변수에서 로드)
S3_CONFIG = {
    "ACCESS_KEY_ID": os.getenv("S3_ACCESS_KEY_ID"),
    "SECRET_ACCESS_KEY": os.getenv("S3_SECRET_ACCESS_KEY"),
    "REGION": os.getenv("S3_REGION", "ap-northeast-2"),
    "ACCESS_POINT_ARN": os.getenv("S3_ACCESS_POINT_ARN"),
    "PREFIX": os.getenv("S3_PREFIX", "skala2/"),
    "BASE_URL": os.getenv("S3_BASE_URL")
}


# S3 클라이언트 싱글톤
_s3_client = None


def get_s3_client():
    """S3 클라이언트 가져오기 (싱글톤)"""
    global _s3_client

    if _s3_client is None:
        # 필수 환경변수 체크
        if not S3_CONFIG["ACCESS_KEY_ID"] or not S3_CONFIG["SECRET_ACCESS_KEY"]:
            raise ValueError("S3_ACCESS_KEY_ID와 S3_SECRET_ACCESS_KEY 환경변수가 필요합니다")

        if not S3_CONFIG["ACCESS_POINT_ARN"]:
            raise ValueError("S3_ACCESS_POINT_ARN 환경변수가 필요합니다")

        _s3_client = boto3.client(
            's3',
            aws_access_key_id=S3_CONFIG["ACCESS_KEY_ID"],
            aws_secret_access_key=S3_CONFIG["SECRET_ACCESS_KEY"],
            region_name=S3_CONFIG["REGION"]
        )

    return _s3_client


def upload_screenshot_to_s3(
    screenshot_bytes: bytes,
    filename: str,
    content_type: str = "image/png"
) -> Optional[str]:
    """
    스크린샷을 S3에 업로드

    Args:
        screenshot_bytes: 스크린샷 바이트 데이터
        filename: 파일명 (예: "toss_job_123.png")
        content_type: MIME 타입 (기본: "image/png")

    Returns:
        업로드된 S3 URL (성공 시) 또는 None (실패 시)

    Example:
        >>> with open("screenshot.png", "rb") as f:
        >>>     url = upload_screenshot_to_s3(f.read(), "toss_job_123.png")
        >>> print(url)
        https://sk-team-storage-881490135253.s3-accesspoint.ap-northeast-2.amazonaws.com/skala2/images/toss_job_123.png
    """
    try:
        s3_client = get_s3_client()

        # S3 키 생성: skala2/images/{filename}
        s3_key = f"{S3_CONFIG['PREFIX']}images/{filename}"

        # S3에 업로드
        response = s3_client.put_object(
            Bucket=S3_CONFIG["ACCESS_POINT_ARN"],
            Key=s3_key,
            Body=screenshot_bytes,
            ContentType=content_type
        )

        # 업로드 성공 시 URL 반환
        url = f"{S3_CONFIG['BASE_URL']}/{s3_key}"

        return url

    except ClientError as e:
        print(f"S3 업로드 실패 (ClientError): {e}")
        return None

    except Exception as e:
        print(f"S3 업로드 실패: {e}")
        return None


def upload_file_to_s3(
    local_path: str,
    s3_key: str,
    content_type: Optional[str] = None
) -> Optional[str]:
    """
    로컬 파일을 S3에 업로드

    Args:
        local_path: 로컬 파일 경로
        s3_key: S3 키 (전체 경로, prefix 포함)
        content_type: MIME 타입 (None이면 자동 추론)

    Returns:
        업로드된 S3 URL (성공 시) 또는 None (실패 시)
    """
    try:
        s3_client = get_s3_client()

        # Content-Type 자동 추론
        if content_type is None:
            if local_path.endswith('.png'):
                content_type = 'image/png'
            elif local_path.endswith('.jpg') or local_path.endswith('.jpeg'):
                content_type = 'image/jpeg'
            elif local_path.endswith('.txt'):
                content_type = 'text/plain'
            else:
                content_type = 'application/octet-stream'

        # 파일 읽기
        with open(local_path, 'rb') as f:
            file_bytes = f.read()

        # S3에 업로드
        response = s3_client.put_object(
            Bucket=S3_CONFIG["ACCESS_POINT_ARN"],
            Key=s3_key,
            Body=file_bytes,
            ContentType=content_type
        )

        # 업로드 성공 시 URL 반환
        url = f"{S3_CONFIG['BASE_URL']}/{s3_key}"

        return url

    except ClientError as e:
        print(f"S3 파일 업로드 실패 (ClientError): {e}")
        return None

    except Exception as e:
        print(f"S3 파일 업로드 실패: {e}")
        return None


def delete_from_s3(s3_key: str) -> bool:
    """
    S3에서 파일 삭제 (필요 시 사용)

    Args:
        s3_key: 삭제할 S3 키

    Returns:
        성공 여부
    """
    try:
        s3_client = get_s3_client()

        s3_client.delete_object(
            Bucket=S3_CONFIG["ACCESS_POINT_ARN"],
            Key=s3_key
        )

        return True

    except Exception as e:
        print(f"S3 삭제 실패: {e}")
        return False


# 사용 예시 (주석)
"""
# 크롤러에서 사용하는 방법:

from app.utils.s3_uploader import upload_screenshot_to_s3

# 방법 1: Playwright page.screenshot()로 바이트 직접 받기
screenshot_bytes = page.screenshot(full_page=True)
url = upload_screenshot_to_s3(screenshot_bytes, "toss_job_123.png")

if url:
    job_info["screenshots"]["combined"] = url
    print(f"S3 업로드 성공: {url}")
else:
    print("S3 업로드 실패")

# 방법 2: 기존 로컬 파일 업로드 (Saramin 등 특수 케이스)
from app.utils.s3_uploader import upload_file_to_s3

url = upload_file_to_s3(
    "../../img/saramin_job_123.png",
    "skala2/images/saramin_job_123.png"
)
"""
