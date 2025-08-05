"""
JSON保存機能のテスト

TDDアプローチでタグ付け結果のJSON保存/読み込み機能をテストする
"""
import pytest
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime
from src.json_handler import save_tagging_results, load_tagging_results


class TestJSONHandler:
    """JSON保存機能のテストクラス"""
    
    def test_save_tagging_results_basic(self):
        """基本的なタグ付け結果の保存をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "url": "http://example1.com",
                "body": "テスト記事1の内容です。",
                "topics": [
                    {
                        "id": 0,
                        "weight": 0.7,
                        "words": ["テスト", "記事", "内容"]
                    }
                ],
                "tags": ["テストタグ1", "記事タグ1"]
            },
            {
                "doc_id": 1,
                "url": "http://example2.com",
                "body": "テスト記事2の内容です。",
                "topics": [
                    {
                        "id": 1,
                        "weight": 0.6,
                        "words": ["テスト", "記事"]
                    }
                ],
                "tags": ["テストタグ2", "記事タグ2"]
            }
        ]
        
        metadata = {
            "lda_params": {
                "num_topics": 8,
                "alpha": 0.1,
                "eta": 0.01
            },
            "llm_model": "openai/gpt-3.5-turbo"
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file_path = f.name
        
        try:
            # Act
            save_tagging_results(documents_data, metadata, temp_file_path)
            
            # Assert - ファイルが作成されていることを確認
            assert Path(temp_file_path).exists()
            
            # JSON内容を検証
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert "metadata" in saved_data
            assert "documents" in saved_data
            assert "timestamp" in saved_data["metadata"]
            assert saved_data["metadata"]["lda_params"]["num_topics"] == 8
            assert saved_data["metadata"]["llm_model"] == "openai/gpt-3.5-turbo"
            assert len(saved_data["documents"]) == 2
            assert saved_data["documents"][0]["doc_id"] == 0
            assert saved_data["documents"][0]["tags"] == ["テストタグ1", "記事タグ1"]
            
        finally:
            Path(temp_file_path).unlink()
            
    def test_load_tagging_results_basic(self):
        """基本的なタグ付け結果の読み込みをテスト"""
        # Arrange - テストデータを作成
        test_data = {
            "metadata": {
                "timestamp": "2023-01-01T12:00:00",
                "lda_params": {
                    "num_topics": 5,
                    "alpha": 0.1,
                    "eta": 0.01
                },
                "llm_model": "openai/gpt-3.5-turbo"
            },
            "documents": [
                {
                    "doc_id": 0,
                    "url": "http://test.com",
                    "body": "テスト内容",
                    "topics": [{"id": 0, "weight": 0.8, "words": ["テスト"]}],
                    "tags": ["テストタグ"]
                }
            ]
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            temp_file_path = f.name
        
        try:
            # Act
            loaded_data = load_tagging_results(temp_file_path)
            
            # Assert
            assert "metadata" in loaded_data
            assert "documents" in loaded_data
            assert loaded_data["metadata"]["lda_params"]["num_topics"] == 5
            assert len(loaded_data["documents"]) == 1
            assert loaded_data["documents"][0]["tags"] == ["テストタグ"]
            
        finally:
            Path(temp_file_path).unlink()
            
    def test_save_tagging_results_with_japanese_characters(self):
        """日本語文字を含むデータの保存をテスト"""
        # Arrange
        documents_data = [
            {
                "doc_id": 0,
                "url": "https://日本語ドメイン.com",
                "body": "これは日本語のテスト記事です。特殊文字：①②③、絵文字：😀🎉",
                "topics": [
                    {
                        "id": 0,
                        "weight": 0.9,
                        "words": ["日本語", "テスト", "記事", "特殊文字", "絵文字"]
                    }
                ],
                "tags": ["日本語タグ", "テスト記事", "特殊文字含有"]
            }
        ]
        
        metadata = {
            "lda_params": {"num_topics": 3},
            "llm_model": "test-model"
        }
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file_path = f.name
        
        try:
            # Act
            save_tagging_results(documents_data, metadata, temp_file_path)
            
            # Assert
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            doc = saved_data["documents"][0]
            assert "日本語のテスト記事です。特殊文字：①②③、絵文字：😀🎉" in doc["body"]
            assert "日本語タグ" in doc["tags"]
            assert "日本語" in doc["topics"][0]["words"]
            
        finally:
            Path(temp_file_path).unlink()
            
    def test_save_tagging_results_nonexistent_directory(self):
        """存在しないディレクトリへの保存エラーハンドリング"""
        # Arrange
        documents_data = [{"doc_id": 0, "tags": ["test"]}]
        metadata = {"test": "data"}
        nonexistent_path = "/nonexistent/directory/file.json"
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            save_tagging_results(documents_data, metadata, nonexistent_path)
            
    def test_load_tagging_results_nonexistent_file(self):
        """存在しないファイルの読み込みエラーハンドリング"""
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="指定されたファイル.*が見つかりませんでした"):
            load_tagging_results("/nonexistent/file.json")
            
    def test_load_tagging_results_invalid_json(self):
        """無効なJSON形式のファイル読み込みエラーハンドリング"""
        # Arrange - 無効なJSONファイルを作成
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_file_path = f.name
        
        try:
            # Act & Assert
            with pytest.raises(json.JSONDecodeError):
                load_tagging_results(temp_file_path)
                
        finally:
            Path(temp_file_path).unlink()
            
    def test_save_tagging_results_empty_data(self):
        """空のデータの保存をテスト"""
        # Arrange
        documents_data = []
        metadata = {"test": "empty"}
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file_path = f.name
        
        try:
            # Act
            save_tagging_results(documents_data, metadata, temp_file_path)
            
            # Assert
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert saved_data["documents"] == []
            assert saved_data["metadata"]["test"] == "empty"
            
        finally:
            Path(temp_file_path).unlink()