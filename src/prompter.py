"""
プロンプト生成機能

script2のmd_prompt_2を参考に、LDAトピック情報と文書本文を基にした
タグ生成用プロンプトを生成する。
"""
from typing import List, Dict, Any


def generate_tagging_prompt(topic_dist: List[Dict[str, Any]], text: str) -> str:
    """
    LDAトピック分布と文書本文からタグ生成用プロンプトを生成する
    
    Args:
        topic_dist: トピック分布情報のリスト
            [{"id": int, "weight": float, "distribution": [{"word": str, "weight": float}]}]
        text: 文書本文
        
    Returns:
        str: LLM用のタグ生成プロンプト
        
    Raises:
        ValueError: トピック情報またはテキストが空の場合
    """
    if not topic_dist:
        raise ValueError("トピック情報が空です")
    
    if not text.strip():
        raise ValueError("テキストが空です")
    
    # トピック情報をフォーマット
    topic_segments = []
    for topic_info in topic_dist:
        topic_id = topic_info["id"]
        weight = topic_info["weight"]
        words = topic_info["distribution"]
        
        # 各トピックの単語リストを作成
        word_list = []
        for word_info in words:
            word = word_info["word"]
            word_weight = word_info["weight"]
            word_list.append(f"{word}({word_weight:.2f})")
        
        topic_segment = f"""トピック番号{topic_id}
本文におけるトピックの構成度合い:{weight:.2f}
トピックの構成要素:{', '.join(word_list)}"""
        
        topic_segments.append(topic_segment)
    
    # 全体のプロンプトを構築
    topics_text = "\n\n".join(topic_segments)
    
    prompt = f"""以下のトピック情報と本文を分析して、この文書を適切に表現するタグを生成してください。

トピックと、付随情報:
{topics_text}

本文:
{text}

指示:
- 上記のトピック情報と本文の内容を総合的に分析してください
- この文書の内容を具体的かつ適切に表現するタグを3個から5個生成してください
- 汎用的すぎるタグではなく、文書固有の内容を反映したタグを作成してください
- 出力はJSON形式で、以下のようにしてください: {{"0": "タグ1", "1": "タグ2", "2": "タグ3"}}
- JSON形式以外の追加説明は不要です"""

    return prompt