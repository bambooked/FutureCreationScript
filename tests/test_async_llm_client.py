"""
非同期LLMクライアントのテスト

TDDアプローチで非同期処理対応のLLMクライアントをテストする
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, Mock
from src.async_llm_client import AsyncLLMClient


class TestAsyncLLMClient:
    """非同期LLMクライアントのテストクラス"""
    
    @pytest.fixture
    def mock_async_client(self):
        """非同期HTTPクライアントのモック"""
        mock_client = AsyncMock()
        mock_response = Mock()  # レスポンスは同期オブジェクト
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"0": "テストタグ1", "1": "テストタグ2", "2": "テストタグ3"}'
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        return mock_client
    
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    def test_async_llm_client_initialization(self):
        """非同期LLMクライアントの初期化をテスト"""
        # Act
        client = AsyncLLMClient()
        
        # Assert
        assert client._AsyncLLMClient__model == "google/gemini-2.0-flash-001"
        assert "Bearer test-api-key" in client._AsyncLLMClient__headers["Authorization"]
        
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    def test_set_model(self):
        """モデル設定のテスト"""
        # Arrange
        client = AsyncLLMClient()
        
        # Act
        client.set_model("openai/gpt-3.5-turbo")
        
        # Assert
        assert client._AsyncLLMClient__model == "openai/gpt-3.5-turbo"
        
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @pytest.mark.asyncio
    async def test_post_basic_message_single(self, mock_async_client):
        """単一メッセージの非同期送信をテスト"""
        # Arrange
        client = AsyncLLMClient()
        messages = [{"role": "user", "content": "テストプロンプト"}]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            
            # Act
            result = await client.post_basic_message(messages)
            
            # Assert
            assert result == '{"0": "テストタグ1", "1": "テストタグ2", "2": "テストタグ3"}'
            mock_async_client.post.assert_called_once()
            
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @pytest.mark.asyncio
    async def test_post_multiple_messages_concurrent(self, mock_async_client):
        """複数メッセージの同時送信をテスト"""
        # Arrange
        client = AsyncLLMClient()
        messages_list = [
            [{"role": "user", "content": "プロンプト1"}],
            [{"role": "user", "content": "プロンプト2"}],
            [{"role": "user", "content": "プロンプト3"}]
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            
            # Act
            results = await client.post_multiple_messages(messages_list)
            
            # Assert
            assert len(results) == 3
            assert all(isinstance(result, str) for result in results)
            assert mock_async_client.post.call_count == 3
            
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @pytest.mark.asyncio
    async def test_post_multiple_messages_with_semaphore(self, mock_async_client):
        """セマフォによる同時接続数制限のテスト"""
        # Arrange
        client = AsyncLLMClient(max_concurrent=2)
        messages_list = [
            [{"role": "user", "content": f"プロンプト{i}"}] for i in range(5)
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            
            # Act
            results = await client.post_multiple_messages(messages_list)
            
            # Assert  
            assert len(results) == 5
            assert mock_async_client.post.call_count == 5
            
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @pytest.mark.asyncio
    async def test_post_message_with_http_error(self, mock_async_client):
        """HTTP エラー時のエラーハンドリングをテスト"""
        # Arrange
        client = AsyncLLMClient()
        messages = [{"role": "user", "content": "テストプロンプト"}]
        
        import httpx
        mock_async_client.post.side_effect = httpx.HTTPStatusError(
            "Client error '401 Unauthorized'", 
            request=Mock(), 
            response=Mock(status_code=401, text="Unauthorized")
        )
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            
            # Act & Assert
            with pytest.raises(httpx.HTTPStatusError):
                await client.post_basic_message(messages)
                
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @pytest.mark.asyncio
    async def test_post_message_with_network_error(self, mock_async_client):
        """ネットワークエラー時のエラーハンドリングをテスト"""
        # Arrange
        client = AsyncLLMClient()
        messages = [{"role": "user", "content": "テストプロンプト"}]
        
        mock_async_client.post.side_effect = Exception("Network error")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            
            # Act & Assert
            with pytest.raises(Exception, match="Network error"):
                await client.post_basic_message(messages)
                
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    @pytest.mark.asyncio
    async def test_post_message_with_metadata(self, mock_async_client):
        """メタデータ付きレスポンスのテスト"""
        # Arrange
        client = AsyncLLMClient()
        messages = [{"role": "user", "content": "テストプロンプト"}]
        
        full_response = {
            "choices": [{"message": {"content": "テスト結果"}}],
            "usage": {"total_tokens": 100}
        }
        mock_async_client.post.return_value.json.return_value = full_response
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_async_client
            
            # Act
            result = await client.post_basic_message(messages, include_meta_data=True)
            
            # Assert
            assert result == full_response
            assert result["usage"]["total_tokens"] == 100
            
    def test_missing_api_key_error(self):
        """API key不足時のエラーハンドリングをテスト"""
        # Arrange & Act & Assert
        with patch.dict('os.environ', {}, clear=True):
            with patch('dotenv.load_dotenv') as mock_load_dotenv:
                mock_load_dotenv.return_value = None
                with pytest.raises(ValueError, match="API key not found"):
                    AsyncLLMClient()
                
    @patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-api-key'})
    def test_default_gemini_flash_model(self):
        """デフォルトでGemini Flashモデルが設定されることをテスト"""
        # Act
        client = AsyncLLMClient()
        
        # Assert
        assert client._AsyncLLMClient__model == "google/gemini-2.0-flash-001"