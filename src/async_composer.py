"""
非同期タグ生成コンポーザー

LDAモデルと非同期LLMClientを統合して、高速な文書タグ生成を行う
"""
import asyncio
import json
from typing import List, Any
from ast import literal_eval

from .lda_model import LDATopicModel
from .async_llm_client import AsyncLLMClient
from .prompter import generate_tagging_prompt


class AsyncTaggingComposer:
    """
    LDAトピックモデルと非同期LLMを組み合わせたタグ生成クラス
    """
    
    def __init__(self, lda_model: LDATopicModel, async_llm_client: AsyncLLMClient):
        """
        初期化
        
        Args:
            lda_model: 学習済みLDAモデル
            async_llm_client: 非同期LLMクライアント
        """
        self.lda_model = lda_model
        self.async_llm_client = async_llm_client
        
    async def generate_tags(self, text: str) -> List[str]:
        """
        指定されたテキストから非同期でタグを生成する
        
        Args:
            text: タグ生成対象のテキスト
            
        Returns:
            List[str]: 生成されたタグのリスト
            
        Raises:
            ValueError: テキストが空の場合
        """
        if not text.strip():
            raise ValueError("テキストが空です")
            
        # LDAでトピック分析（同期処理）
        topic_dist = self.lda_model.analyze_single_text_ad3(
            text,
            topic_max_len=5,
            word_max_len=10,
            topic_weight_threshold=0.0,
            word_weight_threshold=0.0
        )
        
        # プロンプト生成
        prompt = generate_tagging_prompt(topic_dist, text)
        
        # 非同期でLLMにタグ生成を依頼
        response = await self.async_llm_client.post_basic_message([
            {"role": "user", "content": prompt}
        ])
        
        # レスポンスをフォーマット
        tags = self._format_response(response)
        
        return tags
        
    async def generate_multiple_tags(
        self, 
        texts: List[str], 
        batch_size: int = 10,
        rate_limit_delay: float = 0.0
    ) -> List[List[str]]:
        """
        複数のテキストから並行してタグを生成する
        
        Args:
            texts: タグ生成対象のテキストリスト
            batch_size: 並行処理のバッチサイズ
            rate_limit_delay: レート制限のための遅延（秒）
            
        Returns:
            List[List[str]]: 各テキストに対応するタグリストのリスト
        """
        all_tags = []
        
        # バッチ処理で並行実行
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 各テキストのLDA分析とプロンプト生成（同期処理）
            batch_prompts = []
            for text in batch_texts:
                if not text.strip():
                    batch_prompts.append([{"role": "user", "content": "空のテキストです"}])
                    continue
                    
                topic_dist = self.lda_model.analyze_single_text_ad3(
                    text,
                    topic_max_len=5,
                    word_max_len=10,
                    topic_weight_threshold=0.0,
                    word_weight_threshold=0.0
                )
                
                prompt = generate_tagging_prompt(topic_dist, text)
                batch_prompts.append([{"role": "user", "content": prompt}])
            
            # 非同期で並行してLLMに送信
            batch_responses = await self.async_llm_client.post_multiple_messages(batch_prompts)
            
            # レスポンスをフォーマット
            batch_tags = [self._format_response(response) for response in batch_responses]
            all_tags.extend(batch_tags)
            
            # レート制限のための遅延
            if rate_limit_delay > 0 and i + batch_size < len(texts):
                await asyncio.sleep(rate_limit_delay)
        
        return all_tags
        
    def _format_response(self, response: str) -> List[str]:
        """
        LLMからの応答をタグリストにフォーマットする
        
        Args:
            response: LLMからの応答文字列
            
        Returns:
            List[str]: フォーマットされたタグリスト
        """
        try:
            # JSON形式として解析を試行
            formatted_dict = literal_eval(response.strip())
            if isinstance(formatted_dict, dict):
                return list(formatted_dict.values())
            else:
                # 辞書でない場合はフォールバック
                return self._fallback_format(response)
        except (ValueError, SyntaxError):
            # JSON解析失敗時のフォールバック
            return self._fallback_format(response)
            
    def _fallback_format(self, response: str) -> List[str]:
        """
        JSON解析失敗時のフォールバック処理
        
        Args:
            response: 応答文字列
            
        Returns:
            List[str]: カンマ区切りでパースしたタグリスト
        """
        # カンマ区切りとして処理
        tags = [item.strip() for item in response.split(',')]
        # 空文字列を除去
        tags = [tag for tag in tags if tag]
        return tags