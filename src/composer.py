"""
タグ生成コンポーザー

LDAモデルとLLMClientを統合して、文書に対するタグ生成を行う
script2のComposerクラスを参考に実装
"""
import json
from typing import List, Any
from ast import literal_eval

from .lda_model import LDATopicModel
from .llm_client import LLMClient
from .prompter import generate_tagging_prompt


class TaggingComposer:
    """
    LDAトピックモデルとLLMを組み合わせたタグ生成クラス
    """
    
    def __init__(self, lda_model: LDATopicModel, llm_client: LLMClient):
        """
        初期化
        
        Args:
            lda_model: 学習済みLDAモデル
            llm_client: LLMクライアント
        """
        self.lda_model = lda_model
        self.llm_client = llm_client
        
    def generate_tags(self, text: str) -> List[str]:
        """
        指定されたテキストからタグを生成する
        
        Args:
            text: タグ生成対象のテキスト
            
        Returns:
            List[str]: 生成されたタグのリスト
            
        Raises:
            ValueError: テキストが空の場合
        """
        if not text.strip():
            raise ValueError("テキストが空です")
            
        # LDAでトピック分析
        topic_dist = self.lda_model.analyze_single_text_ad3(
            text,
            topic_max_len=5,
            word_max_len=10,
            topic_weight_threshold=0.0,
            word_weight_threshold=0.0
        )
        
        # プロンプト生成
        prompt = generate_tagging_prompt(topic_dist, text)
        
        # LLMでタグ生成
        response = self.llm_client.post_basic_message([
            {"role": "user", "content": prompt}
        ])
        
        # レスポンスをフォーマット
        tags = self._format_response(response)
        
        return tags
        
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