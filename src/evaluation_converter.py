"""
評価用データ変換機能

タグ付け結果をComprehensiveEvaluatorが要求する形式に変換する
"""
from typing import List, Dict, Any, Tuple


def convert_to_evaluation_format(documents_data: List[Dict[str, Any]]) -> Tuple[List[str], Dict[int, Dict[int, str]]]:
    """
    タグ付け結果をComprehensiveEvaluator用の形式に変換する
    
    Args:
        documents_data: タグ付け結果のリスト
            [{"doc_id": int, "body": str, "tags": List[str], ...}]
            
    Returns:
        Tuple[List[str], Dict[int, Dict[int, str]]]:
            - documents: 文書本文のリスト
            - tags_dict: {文書インデックス: {タグインデックス: タグ}} の辞書
            
    Raises:
        KeyError: 必須フィールドが不足している場合
    """
    if not documents_data:
        return [], {}
    
    # doc_idでソートして一貫性を保つ
    sorted_documents = sorted(documents_data, key=lambda x: x.get("doc_id", 0))
    
    documents = []
    tags_dict = {}
    
    for doc_index, doc_data in enumerate(sorted_documents):
        # 必須フィールドの存在確認
        if "body" not in doc_data:
            raise KeyError("必須フィールド 'body' が見つかりません")
        if "tags" not in doc_data:
            raise KeyError("必須フィールド 'tags' が見つかりません")
        
        # 文書本文を追加
        documents.append(doc_data["body"])
        
        # タグを辞書形式に変換
        doc_tags = doc_data["tags"]
        tag_dict = {}
        for tag_index, tag in enumerate(doc_tags):
            tag_dict[tag_index] = tag
        
        tags_dict[doc_index] = tag_dict
    
    return documents, tags_dict