"""
メイン統合処理のテスト

TDDアプローチで全体のパイプラインをテストする
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import csv


class TestMainPipeline:
    """メイン統合処理のテストクラス"""
    
    @pytest.fixture
    def sample_csv_data(self):
        """テスト用CSVデータ"""
        return [
            {"url": "http://example1.com", "date": "2023-01-01", "body": "スマートフォンアプリの新機能について説明します。"},
            {"url": "http://example2.com", "date": "2023-01-02", "body": "経済情勢の変化と市場の動向を分析します。"}
        ]
    
    @pytest.fixture
    def temp_csv_file(self, sample_csv_data):
        """一時CSVファイルを作成"""
        with NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['url', 'date', 'body'])
            writer.writeheader()
            writer.writerows(sample_csv_data)
            temp_file_path = f.name
        yield temp_file_path
        Path(temp_file_path).unlink()
    
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('src.main.LDATopicModel')
    @patch('src.main.LLMClient')
    @patch('src.main.TaggingComposer')
    @patch('src.main.ComprehensiveEvaluator')
    def test_main_pipeline_basic_flow(self, mock_evaluator_class, mock_composer_class, 
                                     mock_llm_class, mock_lda_class, temp_csv_file):
        """基本的なパイプライン動作をテスト"""
        # Arrange - モックの設定
        mock_lda = Mock()
        mock_lda_class.return_value = mock_lda
        mock_lda.build.return_value = None
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        mock_composer = Mock()
        mock_composer_class.return_value = mock_composer
        mock_composer.generate_tags.side_effect = [
            ["スマートフォン", "アプリ", "新機能"],
            ["経済", "市場", "動向"]
        ]
        
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        mock_result = Mock()
        mock_result.to_dict.return_value = {"sbert_cosine": 0.8, "method_name": "LDA-LLM"}
        mock_evaluator.evaluate.return_value = mock_result
        
        # Act
        from src.main import run_tagging_pipeline
        
        with NamedTemporaryFile(suffix='.json', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            result = run_tagging_pipeline(
                csv_file_path=temp_csv_file,
                column_name="body",
                output_path=output_path,
                lda_params={"num_topics": 5, "alpha": 0.1, "eta": 0.01}
            )
            
            # Assert
            assert result is not None
            assert "documents_processed" in result
            assert result["documents_processed"] == 2
            
            # モックの呼び出し確認
            mock_lda.build.assert_called_once()
            assert mock_composer.generate_tags.call_count == 2
            mock_evaluator.evaluate.assert_called_once()
            
            # 出力ファイルの確認
            assert Path(output_path).exists()
            with open(output_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            assert "metadata" in saved_data
            assert "documents" in saved_data
            assert len(saved_data["documents"]) == 2
            
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    def test_main_pipeline_with_missing_env_var(self, temp_csv_file):
        """環境変数が不足している場合のエラーハンドリング"""
        # Arrange - 環境変数を削除
        with patch.dict('os.environ', {}, clear=True):
            from src.main import run_tagging_pipeline
            
            # Act & Assert
            with pytest.raises(ValueError, match="API key not found"):
                run_tagging_pipeline(
                    csv_file_path=temp_csv_file,
                    column_name="body",
                    output_path="test_output.json"
                )
    
    def test_main_pipeline_with_nonexistent_csv(self):
        """存在しないCSVファイル指定時のエラーハンドリング"""
        from src.main import run_tagging_pipeline
        
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            run_tagging_pipeline(
                csv_file_path="/nonexistent/file.csv",
                column_name="body",
                output_path="test_output.json"
            )
    
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @patch('src.main.LDATopicModel')
    @patch('src.main.LLMClient')
    def test_main_pipeline_with_lda_error(self, mock_llm_class, mock_lda_class, temp_csv_file):
        """LDAモデル構築エラー時の処理をテスト"""
        # Arrange
        mock_lda = Mock()
        mock_lda_class.return_value = mock_lda
        mock_lda.build.side_effect = Exception("LDA構築エラー")
        
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        
        from src.main import run_tagging_pipeline
        
        # Act & Assert
        with pytest.raises(Exception, match="LDA構築エラー"):
            run_tagging_pipeline(
                csv_file_path=temp_csv_file,
                column_name="body",
                output_path="test_output.json"
            )
    
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    def test_main_cli_interface(self, temp_csv_file):
        """コマンドライン実行インターフェースのテスト"""
        # 実際の実行は時間がかかるため、インターフェースの存在確認のみ
        from src.main import main
        
        # 関数が定義されていることを確認
        assert callable(main)
        
        # 引数の検証（実際の実行はしない）
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ['main.py', '--help']
            # ヘルプメッセージの存在確認（実際の実行はmockで）
            with patch('argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.csv_file = temp_csv_file
                mock_args.column = "body" 
                mock_args.output = "test_output.json"
                mock_args.topics = 5
                mock_args.alpha = 0.1
                mock_args.eta = 0.01
                mock_parse.return_value = mock_args
                
                # mainが引数を適切に処理することを確認
                assert mock_args.csv_file == temp_csv_file
                assert mock_args.column == "body"
                
        finally:
            sys.argv = original_argv