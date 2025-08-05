"""
Composerクラスのテスト

TDDアプローチでLDAモデルとLLMClientの統合機能をテストする
"""
import pytest
from unittest.mock import Mock, patch
import json
from src.composer import TaggingComposer


class TestTaggingComposer:
    """タグ生成コンポーザーのテストクラス"""
    
    @pytest.fixture
    def mock_lda_model(self):
        """LDAモデルのモック"""
        mock = Mock()
        mock.analyze_single_text_ad3.return_value = [
            {
                "id": 0,
                "weight": 0.7,
                "distribution": [
                    {"word": "スマートフォン", "weight": 0.3},
                    {"word": "アプリ", "weight": 0.25}
                ]
            }
        ]
        return mock
        
    @pytest.fixture
    def mock_llm_client(self):
        """LLMクライアントのモック"""
        mock = Mock()
        mock.post_basic_message.return_value = '{"0": "スマートフォンアプリ", "1": "モバイル技術", "2": "デジタル革新"}'
        return mock
        
    @pytest.fixture
    def composer(self, mock_lda_model, mock_llm_client):
        """Composerインスタンス"""
        return TaggingComposer(mock_lda_model, mock_llm_client)
        
    def test_composer_initialization(self, mock_lda_model, mock_llm_client):
        """Composerの初期化をテスト"""
        # Act
        composer = TaggingComposer(mock_lda_model, mock_llm_client)
        
        # Assert
        assert composer.lda_model == mock_lda_model
        assert composer.llm_client == mock_llm_client
        
    def test_generate_tags_for_single_document(self, composer, mock_lda_model, mock_llm_client):
        """単一文書のタグ生成をテスト"""
        # Arrange
        text = "新しいスマートフォンアプリが登場しました。"
        
        # Act
        tags = composer.generate_tags(text)
        
        # Assert
        assert isinstance(tags, list)
        assert len(tags) == 3
        assert "スマートフォンアプリ" in tags
        assert "モバイル技術" in tags
        assert "デジタル革新" in tags
        
        # モックの呼び出し確認
        mock_lda_model.analyze_single_text_ad3.assert_called_once_with(
            text, 
            topic_max_len=5,
            word_max_len=10,
            topic_weight_threshold=0.0,
            word_weight_threshold=0.0
        )
        mock_llm_client.post_basic_message.assert_called_once()
        
    def test_generate_tags_with_malformed_json_response(self, composer, mock_lda_model, mock_llm_client):
        """JSONでない応答のフォールバック処理をテスト"""
        # Arrange
        text = "テスト文書です。"
        mock_llm_client.post_basic_message.return_value = "タグ1, タグ2, タグ3"
        
        # Act
        tags = composer.generate_tags(text)
        
        # Assert
        assert isinstance(tags, list)
        assert len(tags) == 3
        assert "タグ1" in tags
        assert "タグ2" in tags
        assert "タグ3" in tags
        
    def test_generate_tags_with_empty_text(self, composer):
        """空のテキストでのエラーハンドリング"""
        # Act & Assert
        with pytest.raises(ValueError, match="テキストが空です"):
            composer.generate_tags("")
            
    @patch('src.composer.generate_tagging_prompt')
    def test_generate_tags_uses_prompter(self, mock_prompt_func, composer, mock_lda_model, mock_llm_client):
        """プロンプト関数が正しく使用されることをテスト"""
        # Arrange
        text = "テスト文書です。"
        mock_prompt_func.return_value = "テストプロンプト"
        
        # Act
        composer.generate_tags(text)
        
        # Assert
        mock_prompt_func.assert_called_once()
        call_args = mock_prompt_func.call_args
        assert call_args[0][1] == text  # 第二引数がテキスト
        
        # LLMに正しいプロンプトが渡されることを確認
        mock_llm_client.post_basic_message.assert_called_once_with([
            {"role": "user", "content": "テストプロンプト"}
        ])
        
    def test_generate_tags_with_lda_error(self, composer, mock_lda_model, mock_llm_client):
        """LDAモデルエラー時の処理をテスト"""
        # Arrange
        text = "テスト文書です。"
        mock_lda_model.analyze_single_text_ad3.side_effect = Exception("LDA処理エラー")
        
        # Act & Assert
        with pytest.raises(Exception, match="LDA処理エラー"):
            composer.generate_tags(text)
            
    def test_generate_tags_with_llm_error(self, composer, mock_lda_model, mock_llm_client):
        """LLMエラー時の処理をテスト"""
        # Arrange
        text = "テスト文書です。"
        mock_llm_client.post_basic_message.side_effect = Exception("LLM通信エラー")
        
        # Act & Assert
        with pytest.raises(Exception, match="LLM通信エラー"):
            composer.generate_tags(text)
            
    def test_format_response_valid_json(self, composer):
        """有効なJSON応答のフォーマットをテスト"""
        # Arrange
        response = '{"0": "タグ1", "1": "タグ2", "2": "タグ3"}'
        
        # Act
        tags = composer._format_response(response)
        
        # Assert
        assert isinstance(tags, list)
        assert len(tags) == 3
        assert "タグ1" in tags
        assert "タグ2" in tags
        assert "タグ3" in tags
        
    def test_format_response_comma_separated(self, composer):
        """カンマ区切り応答のフォーマットをテスト"""
        # Arrange
        response = "タグ1, タグ2, タグ3"
        
        # Act
        tags = composer._format_response(response)
        
        # Assert
        assert isinstance(tags, list)
        assert len(tags) == 3
        assert "タグ1" in tags
        assert "タグ2" in tags
        assert "タグ3" in tags