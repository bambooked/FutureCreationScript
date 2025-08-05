"""
データローダーのテスト

TDDアプローチでCSV読み込み機能をテストする
"""
import pytest
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.data_loader import read_csv_column


class TestDataLoader:
    """データローダーのテストクラス"""
    
    def test_read_csv_column_valid_file(self):
        """正常なCSVファイルの列読み込みをテスト"""
        # Arrange - 一時CSVファイルを作成
        test_data = [
            {"url": "http://example1.com", "date": "2023-01-01", "body": "テスト記事1"},
            {"url": "http://example2.com", "date": "2023-01-02", "body": "テスト記事2"},
            {"url": "http://example3.com", "date": "2023-01-03", "body": "テスト記事3"}
        ]
        
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'date', 'body'])
            writer.writeheader()
            writer.writerows(test_data)
            temp_file_path = f.name
        
        try:
            # Act
            body_column = read_csv_column(temp_file_path, "body")
            
            # Assert
            assert isinstance(body_column, list)
            assert len(body_column) == 3
            assert "テスト記事1" in body_column
            assert "テスト記事2" in body_column
            assert "テスト記事3" in body_column
            
        finally:
            # Cleanup
            Path(temp_file_path).unlink()
            
    def test_read_csv_column_different_column(self):
        """異なる列の読み込みをテスト"""
        # Arrange
        test_data = [
            {"url": "http://example1.com", "date": "2023-01-01", "body": "テスト記事1"},
            {"url": "http://example2.com", "date": "2023-01-02", "body": "テスト記事2"}
        ]
        
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'date', 'body'])
            writer.writeheader()
            writer.writerows(test_data)
            temp_file_path = f.name
        
        try:
            # Act
            url_column = read_csv_column(temp_file_path, "url")
            
            # Assert
            assert len(url_column) == 2
            assert "http://example1.com" in url_column
            assert "http://example2.com" in url_column
            
        finally:
            Path(temp_file_path).unlink()
            
    def test_read_csv_column_nonexistent_file(self):
        """存在しないファイルでのエラーハンドリング"""
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="指定されたファイル.*が見つかりませんでした"):
            read_csv_column("/nonexistent/path/file.csv", "body")
            
    def test_read_csv_column_nonexistent_column(self):
        """存在しない列名でのエラーハンドリング"""
        # Arrange
        test_data = [
            {"url": "http://example1.com", "date": "2023-01-01", "body": "テスト記事1"}
        ]
        
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'date', 'body'])
            writer.writeheader()
            writer.writerows(test_data)
            temp_file_path = f.name
        
        try:
            # Act & Assert
            with pytest.raises(KeyError, match="指定された列名.*がCSVファイルに存在しません"):
                read_csv_column(temp_file_path, "nonexistent_column")
                
        finally:
            Path(temp_file_path).unlink()
            
    def test_read_csv_column_empty_file(self):
        """空のCSVファイルの処理をテスト"""
        # Arrange
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'date', 'body'])
            writer.writeheader()
            # データは書き込まない
            temp_file_path = f.name
        
        try:
            # Act
            body_column = read_csv_column(temp_file_path, "body")
            
            # Assert
            assert isinstance(body_column, list)
            assert len(body_column) == 0
            
        finally:
            Path(temp_file_path).unlink()
            
    def test_read_csv_column_with_encoding_issues(self):
        """エンコーディングの問題がある場合のテスト"""
        # Arrange - 日本語文字を含むデータ
        test_data = [
            {"url": "http://example1.com", "date": "2023-01-01", "body": "日本語のテスト記事です。特殊文字：①②③"},
            {"url": "http://example2.com", "date": "2023-01-02", "body": "もう一つの日本語記事。絵文字：😀🎉"}
        ]
        
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'date', 'body'])
            writer.writeheader()
            writer.writerows(test_data)
            temp_file_path = f.name
        
        try:
            # Act
            body_column = read_csv_column(temp_file_path, "body")
            
            # Assert
            assert len(body_column) == 2
            assert "日本語のテスト記事です。特殊文字：①②③" in body_column
            assert "もう一つの日本語記事。絵文字：😀🎉" in body_column
            
        finally:
            Path(temp_file_path).unlink()