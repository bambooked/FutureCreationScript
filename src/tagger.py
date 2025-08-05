"""
新手法（NM: New Method）によるタグ生成モジュール

このモジュールは、トピックモデルとLLMを組み合わせた文書タグ生成の新手法を実装しています。
従来のトピックモデルベース手法とは異なり、トピック情報と原文の両方を考慮して、
より文脈に即したタグを生成することを目的としています。
"""

from typing import List, Dict, Tuple
import json
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TopicAnalysisResult:
    """トピック分析の結果を格納するデータクラス"""
    topic_distribution: List[Tuple[int, float]]  # [(トピック番号, 構成度), ...]
    top_topics: List[Tuple[int, List[str]]]     # [(トピック番号, [単語リスト]), ...]
    original_text: str                           # 分析対象の原文


@dataclass
class TaggingResult:
    """タグ生成結果を格納するデータクラス"""
    tags: Dict[int, str]         # {トピック番号: タグ}
    raw_response: str            # LLMの生の応答
    method: str = "new_method"   # 使用した手法


class LLMInterface(ABC):
    """LLMクライアントのインターフェース"""
    
    @abstractmethod
    def ask(self, prompt: str) -> str:
        """LLMにプロンプトを送信して応答を取得"""
        pass


class TopicModelInterface(ABC):
    """トピックモデルのインターフェース"""
    
    @abstractmethod
    def analyze_single_text(self, text: str) -> Tuple[List[Tuple[int, float]], List[Tuple[int, List[str]]]]:
        """単一文書のトピック分析を実行"""
        pass


class NewMethodTagger:
    """
    新手法によるタグ生成クラス
    
    トピックモデルの分析結果と原文の両方を考慮して、
    文脈に即したタグを生成します。
    """
    
    def __init__(self, topic_model: TopicModelInterface, llm_client: LLMInterface):
        """
        Args:
            topic_model: トピック分析を行うモデル
            llm_client: タグ生成に使用するLLMクライアント
        """
        self.topic_model = topic_model
        self.llm_client = llm_client
    
    def generate_tags(self, text: str, top_n_topics: int = 5) -> TaggingResult:
        """
        文書に対してタグを生成
        
        Args:
            text: タグ付け対象の文書
            top_n_topics: 考慮するトピック数（デフォルト: 5）
            
        Returns:
            TaggingResult: 生成されたタグと関連情報
        """
        # 1. トピック分析を実行
        analysis_result = self._analyze_document(text)
        
        # 2. プロンプトを生成
        prompt = self._create_prompt(analysis_result, top_n_topics)
        
        # 3. LLMに問い合わせ
        llm_response = self.llm_client.ask(prompt)
        
        # 4. レスポンスを解析
        tags = self._parse_response(llm_response)
        
        return TaggingResult(
            tags=tags,
            raw_response=llm_response,
            method="new_method"
        )
    
    def _analyze_document(self, text: str) -> TopicAnalysisResult:
        """文書のトピック分析を実行"""
        topic_dist, top_topics = self.topic_model.analyze_single_text(text)
        
        return TopicAnalysisResult(
            topic_distribution=topic_dist,
            top_topics=top_topics,
            original_text=text
        )
    
    def _create_prompt(self, analysis: TopicAnalysisResult, top_n_topics: int) -> str:
        """
        新手法用のプロンプトを生成
        
        トピック情報と原文の両方を含み、文脈に即したタグ生成を指示します。
        """
        prompt_parts = []
        
        # ヘッダー
        prompt_parts.append("# 文書のタグ生成タスク\n")
        prompt_parts.append("与えられたトピック情報と文書内容を基に、適切なタグを生成してください。\n")
        
        # トピック情報
        prompt_parts.append("## トピック情報\n")
        for i, ((topic_idx, prob), (_, words)) in enumerate(
            zip(analysis.topic_distribution[:top_n_topics], 
                analysis.top_topics[:top_n_topics])
        ):
            prompt_parts.append(f"### トピック {topic_idx}")
            prompt_parts.append(f"- 文書内での構成度: {prob:.3f}")
            prompt_parts.append(f"- トピックを構成する単語: {', '.join(words[:10])}")
            prompt_parts.append("")
        
        # 文書内容
        prompt_parts.append("## 分析対象の文書\n")
        prompt_parts.append(f"```\n{analysis.original_text}\n```\n")
        
        # 指示
        prompt_parts.append("## 指示")
        prompt_parts.append("1. 上記のトピック情報を参考にしつつ、文書の実際の内容を分析してください。")
        prompt_parts.append("2. 各トピックの視点から文書を見た際に最も適切なタグを生成してください。")
        prompt_parts.append("3. **重要**: 汎化されたタグではなく、この文書固有の内容を反映したタグを生成してください。")
        prompt_parts.append("4. 出力形式: JSON形式で {トピック番号: \"タグ\"} として返してください。")
        prompt_parts.append("\n例: {0: \"技術革新\", 1: \"環境問題\", 2: \"経済政策\"}")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(self, response: str) -> Dict[int, str]:
        """
        LLMのレスポンスを解析してタグ辞書を生成
        
        JSON形式またはカンマ区切り形式の両方に対応
        """
        response = response.strip()
        
        # JSON形式の解析を試みる
        try:
            # JSONブロックを抽出
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                json_str = json_match.group(0)
                # 文字列キーを数値キーに変換
                parsed = json.loads(json_str)
                return {int(k): v for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError):
            pass
        
        # カンマ区切り形式の解析を試みる
        tags = {}
        if ',' in response:
            parts = response.split(',')
            for i, part in enumerate(parts):
                tag = part.strip().strip('"\'')
                if tag:
                    tags[i] = tag
        
        # 単一のタグの場合
        if not tags:
            cleaned = response.strip().strip('"\'')
            if cleaned:
                tags[0] = cleaned
        
        return tags


class SimplifiedNewMethodTagger:
    """
    簡略版の新手法タガー
    
    既存のLDATopicModelとLLMClientを直接使用できるバージョン
    """
    
    def __init__(self, lda_model, llm_client):
        """
        Args:
            lda_model: LDATopicModelのインスタンス
            llm_client: LLMClientのインスタンス
        """
        self.lda_model = lda_model
        self.llm_client = llm_client
    
    def tag_document(self, text: str) -> Dict[int, str]:
        """
        文書にタグを付ける（簡略版）
        
        Args:
            text: タグ付け対象の文書
            
        Returns:
            {トピック番号: タグ} の辞書
        """
        # トピック分析
        try:
            result = self.lda_model.analyze_single_text(text)
            
            if isinstance(result, tuple) and len(result) == 2:
                topic_dist, top_topics = result
            else:
                print(f"Unexpected result format: {result}")
                return {0: "エラー: 予期しない形式"}
        except Exception as e:
            print(f"Error in analyze_single_text: {e}")
            return {0: f"エラー: {str(e)}"}
        
        # プロンプト生成（既存のmd_prompt_2と同等）
        prompt = self._generate_prompt(topic_dist, top_topics, text)
        
        # LLMに問い合わせ
        response = self.llm_client.post_basic_message([{"role": "user", "content": prompt}])
        
        # レスポンス解析
        return self._format_response(response)
    
    def _generate_prompt(self, topic_dist, top_topics, text):
        """md_prompt_2と同等のプロンプトを生成"""
        prompt = "# Please analyze the document below and generate appropriate tags.\n\n"
        
        # トピック情報
        for i, (topic_idx, prob) in enumerate(topic_dist[:5]):
            words = top_topics[i] if i < len(top_topics) else []
            prompt += f"## Topic {topic_idx}\n"
            prompt += f"- Degree of composition in the document: {prob:.3f}\n"
            prompt += f"- Topic components: {', '.join(words[:10])}\n\n"
        
        # 文書
        prompt += "## Document content\n"
        prompt += f"```\n{text}\n```\n\n"
        
        # 指示
        prompt += "Based on the above information, generate appropriate tags that reflect "
        prompt += "the specific content of this document when analyzed from the perspective "
        prompt += "of the given topics. Do NOT generate generalized tags.\n"
        prompt += "Output format: {topic_number: \"tag\"}\n"
        
        return prompt
    
    def _format_response(self, response):
        """_format_responseと同等の処理"""
        try:
            # JSON形式を試す
            match = re.search(r'\{[^}]+\}', response)
            if match:
                result = json.loads(match.group(0))
                return {int(k): v for k, v in result.items()}
        except:
            pass
        
        # カンマ区切りを試す
        tags = {}
        if ',' in response:
            for i, tag in enumerate(response.split(',')):
                tags[i] = tag.strip().strip('"\'')
        else:
            tags[0] = response.strip().strip('"\'')
        
        return tags


# 使用例
if __name__ == "__main__":
    # このファイルを直接実行した場合の動作例
    print("新手法タグ生成モジュール")
    print("=" * 50)
    print("このモジュールは以下の機能を提供します：")
    print("1. NewMethodTagger: インターフェースベースの実装")
    print("2. SimplifiedNewMethodTagger: 既存クラスとの互換性を重視した実装")
    print("\n使用方法:")
    print("from new_method_tagger import SimplifiedNewMethodTagger")
    print("tagger = SimplifiedNewMethodTagger(lda_model, llm_client)")
    print("tags = tagger.tag_document('文書内容...')")