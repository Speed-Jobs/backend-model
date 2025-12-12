from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from typing import List, Optional
from app.core.config import settings
import numpy as np


class Embedder:
    """텍스트 임베딩 생성 (Hugging Face Transformers 직접 사용)"""
    
    _model: Optional[AutoModel] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _device: str = None
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """임베딩 모델 로드"""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            print(f"Device: {self._device}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            
            print(f"Embedding model loaded")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 토큰 임베딩의 평균"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def embed(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        if not text:
            return []
        
        with torch.no_grad():
            # 토큰화
            encoded_input = self._tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self._device)
            
            # 모델 실행
            model_output = self._model(**encoded_input)
            
            # Mean pooling
            embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 정규화
            embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()[0].tolist()
    
    async def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        배치 텍스트 임베딩
        
        Args:
            texts: 임베딩할 텍스트 리스트
            batch_size: 배치 크기
        
        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []
        
        # 빈 텍스트 필터링
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            return []
        
        print(f"🔢 Embedding {len(valid_texts)} texts...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                # 토큰화
                encoded_input = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self._device)
                
                # 모델 실행
                model_output = self._model(**encoded_input)
                
                # Mean pooling
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # 정규화
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    print(f"  Progress: {min(i + batch_size, len(valid_texts))}/{len(valid_texts)}")
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        print(f"Embeddings generated")
        return all_embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        # 더미 텍스트로 차원 확인
        with torch.no_grad():
            encoded_input = self._tokenizer(
                "test",
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self._device)
            
            model_output = self._model(**encoded_input)
            embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            return embedding.shape[1]

