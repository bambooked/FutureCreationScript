"""
プロンプター関数のテスト

TDDアプローチでプロンプト生成機能をテストする
"""
import pytest
from src.prompter import generate_tagging_prompt


class TestPrompter:
    """プロンプト生成機能のテストクラス"""
    
    def test_generate_tagging_prompt_with_single_topic(self):
        """単一トピックのタグ生成プロンプトをテスト"""
        # Arrange
        topic_dist = [
            {
                "id": 0,
                "weight": 0.8,
                "distribution": [
                    {"word": "スマートフォン", "weight": 0.3},
                    {"word": "アプリ", "weight": 0.25},
                    {"word": "技術", "weight": 0.2}
                ]
            }
        ]
        text = "新しいスマートフォンアプリが登場しました。この技術は革新的です。"
        
        # Act
        prompt = generate_tagging_prompt(topic_dist, text)
        
        # Assert
        assert "トピック番号0" in prompt
        assert "0.8" in prompt
        assert "スマートフォン" in prompt
        assert "アプリ" in prompt
        assert "技術" in prompt
        assert text in prompt
        assert "JSON" in prompt
        assert "{" in prompt
        
    def test_generate_tagging_prompt_with_multiple_topics(self):
        """複数トピックのタグ生成プロンプトをテスト"""
        # Arrange
        topic_dist = [
            {
                "id": 0,
                "weight": 0.5,
                "distribution": [
                    {"word": "経済", "weight": 0.4},
                    {"word": "市場", "weight": 0.3}
                ]
            },
            {
                "id": 1,
                "weight": 0.3,
                "distribution": [
                    {"word": "政治", "weight": 0.5},
                    {"word": "選挙", "weight": 0.2}
                ]
            }
        ]
        text = "経済市場の動向と政治的な選挙について分析します。"
        
        # Act
        prompt = generate_tagging_prompt(topic_dist, text)
        
        # Assert
        assert "トピック番号0" in prompt
        assert "トピック番号1" in prompt
        assert "0.5" in prompt
        assert "0.3" in prompt
        assert "経済" in prompt
        assert "政治" in prompt
        assert text in prompt
        
    def test_generate_tagging_prompt_with_empty_topics(self):
        """空のトピック情報でのエラーハンドリングをテスト"""
        # Arrange
        topic_dist = []
        text = "テスト文書です。"
        
        # Act & Assert
        with pytest.raises(ValueError, match="トピック情報が空です"):
            generate_tagging_prompt(topic_dist, text)
            
    def test_generate_tagging_prompt_with_empty_text(self):
        """空のテキストでのエラーハンドリングをテスト"""
        # Arrange
        topic_dist = [
            {
                "id": 0,
                "weight": 0.8,
                "distribution": [{"word": "テスト", "weight": 0.5}]
            }
        ]
        text = ""
        
        # Act & Assert
        with pytest.raises(ValueError, match="テキストが空です"):
            generate_tagging_prompt(topic_dist, text)
            
    def test_generate_tagging_prompt_output_format(self):
        """出力形式の指示が適切に含まれているかテスト"""
        # Arrange
        topic_dist = [
            {
                "id": 0,
                "weight": 0.7,
                "distribution": [{"word": "テスト", "weight": 0.5}]
            }
        ]
        text = "テスト文書です。"
        
        # Act
        prompt = generate_tagging_prompt(topic_dist, text)
        
        # Assert
        assert "JSON形式" in prompt
        assert "{0:" in prompt or '{"0":' in prompt
        assert "タグ" in prompt
        assert "3個から5個" in prompt or "3-5個" in prompt