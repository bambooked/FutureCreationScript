"""
非同期LLMクライアント

高速化のためにasync/awaitを使用してLLMとの通信を非同期で行う
Gemini Flash 1.5をデフォルトモデルとして使用
"""
import httpx
import os
import asyncio
from typing import List, Dict, Any, Union


class AsyncLLMClient:
    """非同期LLMクライアント"""
    
    def __init__(self, max_concurrent: int = 10) -> None:
        """
        初期化
        
        Args:
            max_concurrent: 最大同時接続数
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        
        self.__headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.__target_url = "https://openrouter.ai/api/v1/chat/completions"
        self.__model = "google/gemini-2.0-flash-001"  # デフォルトはGemini Flash
        
        # 同時接続数を制限するセマフォ
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
    def set_model(self, model_name: str) -> None:
        """
        使用するLLMモデルを設定
        
        Args:
            model_name: モデル名
        """
        self.__model = model_name
        
    async def post_basic_message(
        self,
        messages: List[Dict[str, str]],
        include_meta_data: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        単一メッセージを非同期で送信
        
        Args:
            messages: メッセージリスト
            include_meta_data: メタデータを含むかどうか
            
        Returns:
            str | Dict[str, Any]: LLMからの応答
        """
        data = {"model": self.__model, "messages": messages}
        
        async with self._semaphore:  # セマフォで同時接続数を制限
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.__target_url, 
                        headers=self.__headers, 
                        json=data
                    )
                response.raise_for_status()
                response_data = response.json()
                
                if not include_meta_data:
                    response_data = response_data["choices"][0]["message"]["content"]
                    
                return response_data
                
            except httpx.HTTPStatusError as e:
                print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise
                
    async def post_multiple_messages(
        self,
        messages_list: List[List[Dict[str, str]]],
        include_meta_data: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        複数メッセージを並行して非同期送信
        
        Args:
            messages_list: メッセージリストのリスト
            include_meta_data: メタデータを含むかどうか
            
        Returns:
            List[Union[str, Dict[str, Any]]]: LLMからの応答リスト
        """
        tasks = [
            self.post_basic_message(messages, include_meta_data)
            for messages in messages_list
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 例外が発生した場合はエラーメッセージに置き換え
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error in message {i}: {str(result)}")
                processed_results.append("")  # エラー時は空文字列
            else:
                processed_results.append(result)
                
        return processed_results