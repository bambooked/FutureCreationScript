import httpx
import os

class LLMClient:
    def __init__(self) -> None:
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
        self.set_target_url()
        self.set_model()

    def set_target_url(self, target_url: str = None) -> None:
        self.__target_url = target_url if target_url else "https://openrouter.ai/api/v1/chat/completions"

    def set_model(self, model_name: str = None) -> None:
        self.__model = model_name if model_name else "openai/gpt-3.5-turbo"

    def post_basic_message(
        self,
        messages: list[dict[str, str]],
        include_meta_data: bool = False
        ) -> str | dict:
        data = {"model": self.__model, "messages": messages}
        try:
            with httpx.Client() as client:
                response = client.post(
                    self.__target_url, headers=self.__headers, json=data
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

if __name__ == "__main__":
    client = LLMClient()
    client.set_model("meta-llama/llama-3.1-70b-instruct")
    prompt = "こんにちは"
    response = client.post_basic_message([{"role":"user", "content":prompt}])
    print(response)