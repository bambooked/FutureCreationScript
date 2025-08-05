"""
評価用データ変換機能のテスト

TDDアプローチでComprehensiveEvaluator用のデータ変換機能をテストする
"""
import pytest
from src.evaluation_converter import convert_to_evaluation_format


class TestEvaluationConverter:
    """評価用データ変換のテストクラス"""
    
    def test_convert_to_evaluation_format_basic(self):
        """基本的なデータ変換をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "url": "http://example1.com",
                "body": "記事1の内容です。",
                "topics": [
                    {"id": 0, "weight": 0.7, "words": ["記事", "内容"]},
                    {"id": 1, "weight": 0.3, "words": ["テスト", "データ"]}
                ],
                "tags": ["記事タグ1", "内容タグ1", "情報タグ1"]
            },
            {
                "doc_id": 1,
                "url": "http://example2.com",
                "body": "記事2の内容です。",
                "topics": [
                    {"id": 0, "weight": 0.6, "words": ["記事", "内容"]},
                    {"id": 2, "weight": 0.4, "words": ["サンプル", "テキスト"]}
                ],
                "tags": ["記事タグ2", "内容タグ2"]
            }
        ]
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert - 文書リストの検証
        assert isinstance(documents, list)
        assert len(documents) == 2
        assert documents[0] == "記事1の内容です。"
        assert documents[1] == "記事2の内容です。"
        
        # Assert - タグ辞書の検証
        assert isinstance(tags_dict, dict)
        assert len(tags_dict) == 2
        
        # 文書0のタグ検証
        assert 0 in tags_dict
        assert len(tags_dict[0]) == 3
        assert tags_dict[0][0] == "記事タグ1"
        assert tags_dict[0][1] == "内容タグ1"
        assert tags_dict[0][2] == "情報タグ1"
        
        # 文書1のタグ検証
        assert 1 in tags_dict
        assert len(tags_dict[1]) == 2
        assert tags_dict[1][0] == "記事タグ2"
        assert tags_dict[1][1] == "内容タグ2"
        
    def test_convert_to_evaluation_format_empty_tags(self):
        """空のタグを持つ文書の変換をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "url": "http://example.com",
                "body": "タグのない記事です。",
                "topics": [{"id": 0, "weight": 0.5, "words": ["記事"]}],
                "tags": []
            }
        ]
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert
        assert len(documents) == 1
        assert documents[0] == "タグのない記事です。"
        assert 0 in tags_dict
        assert len(tags_dict[0]) == 0
        
    def test_convert_to_evaluation_format_single_tag(self):
        """単一タグを持つ文書の変換をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "body": "単一タグの記事です。",
                "tags": ["単一タグ"]
            }
        ]
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert
        assert len(documents) == 1
        assert 0 in tags_dict
        assert len(tags_dict[0]) == 1
        assert tags_dict[0][0] == "単一タグ"
        
    def test_convert_to_evaluation_format_many_tags(self):
        """多数のタグを持つ文書の変換をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "body": "多数のタグを持つ記事です。",
                "tags": ["タグ1", "タグ2", "タグ3", "タグ4", "タグ5", "タグ6", "タグ7"]
            }
        ]
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert
        assert len(documents) == 1
        assert 0 in tags_dict
        assert len(tags_dict[0]) == 7
        for i in range(7):
            assert tags_dict[0][i] == f"タグ{i+1}"
            
    def test_convert_to_evaluation_format_non_sequential_doc_ids(self):
        """非連続のdoc_idを持つデータの変換をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 5,
                "body": "文書5です。",
                "tags": ["タグ5"]
            },
            {
                "doc_id": 2,
                "body": "文書2です。",
                "tags": ["タグ2"]
            },
            {
                "doc_id": 8,
                "body": "文書8です。",
                "tags": ["タグ8"]
            }
        ]
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert - doc_idの順序でソートされていることを確認
        assert len(documents) == 3
        assert documents[0] == "文書2です。"  # doc_id=2が最初
        assert documents[1] == "文書5です。"  # doc_id=5が2番目
        assert documents[2] == "文書8です。"  # doc_id=8が最後
        
        # tags_dictはdoc_idではなく、ソート後のインデックスを使用
        assert 0 in tags_dict  # 元のdoc_id=2 → インデックス0
        assert 1 in tags_dict  # 元のdoc_id=5 → インデックス1
        assert 2 in tags_dict  # 元のdoc_id=8 → インデックス2
        assert tags_dict[0][0] == "タグ2"
        assert tags_dict[1][0] == "タグ5"
        assert tags_dict[2][0] == "タグ8"
        
    def test_convert_to_evaluation_format_empty_input(self):
        """空の入力データの変換をテスト"""
        # Arrange
        documents_data = []
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert
        assert isinstance(documents, list)
        assert len(documents) == 0
        assert isinstance(tags_dict, dict)
        assert len(tags_dict) == 0
        
    def test_convert_to_evaluation_format_missing_fields(self):
        """必須フィールドが不足している場合のエラーハンドリング"""
        # Arrange - bodyフィールドがない
        documents_data = [
            {
                "doc_id": 0,
                "url": "http://example.com",
                "tags": ["タグ1"]
            }
        ]
        
        # Act & Assert
        with pytest.raises(KeyError, match="body"):
            convert_to_evaluation_format(documents_data)
            
    def test_convert_to_evaluation_format_missing_tags_field(self):
        """tagsフィールドが不足している場合のエラーハンドリング"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "body": "記事内容です。"
            }
        ]
        
        # Act & Assert
        with pytest.raises(KeyError, match="tags"):
            convert_to_evaluation_format(documents_data)
            
    def test_convert_to_evaluation_format_japanese_content(self):
        """日本語文書とタグの変換をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "body": "これは日本語の記事です。特殊文字：①②③、絵文字：😀🎉も含みます。",
                "tags": ["日本語", "特殊文字", "絵文字"]
            }
        ]
        
        # Act
        documents, tags_dict = convert_to_evaluation_format(documents_data)
        
        # Assert
        assert documents[0] == "これは日本語の記事です。特殊文字：①②③、絵文字：😀🎉も含みます。"
        assert tags_dict[0][0] == "日本語"
        assert tags_dict[0][1] == "特殊文字"
        assert tags_dict[0][2] == "絵文字"