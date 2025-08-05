"""
JSON保存・読み込み機能

タグ付け結果をJSON形式で保存・読み込みする機能
"""
import json
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """
    Numpyデータ型をJSONシリアライズ可能な型に変換するカスタムエンコーダー
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_tagging_results(
    documents_data: List[Dict[str, Any]], 
    metadata: Dict[str, Any], 
    file_path: str
) -> None:
    """
    タグ付け結果をJSON形式で保存する
    
    Args:
        documents_data: 文書とタグ付け結果のリスト
        metadata: メタデータ（LDAパラメータ、LLMモデル等）
        file_path: 保存先ファイルパス
        
    Raises:
        FileNotFoundError: 保存先ディレクトリが存在しない場合
    """
    # タイムスタンプを追加
    metadata_with_timestamp = metadata.copy()
    metadata_with_timestamp["timestamp"] = datetime.now().isoformat()
    
    # 全体のデータ構造を構築
    output_data = {
        "metadata": metadata_with_timestamp,
        "documents": documents_data
    }
    
    try:
        # ディレクトリが存在しない場合は作成
        file_path_obj = Path(file_path)
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # JSON形式で保存（Numpyエンコーダー使用）
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
    except (OSError, PermissionError) as e:
        if "Read-only file system" in str(e) or "Permission denied" in str(e):
            raise FileNotFoundError(f"保存先ディレクトリが作成できませんでした: {file_path}")
        raise e
    except Exception as e:
        if "No such file or directory" in str(e):
            raise FileNotFoundError(f"保存先ディレクトリが作成できませんでした: {file_path}")
        raise e


def load_tagging_results(file_path: str) -> Dict[str, Any]:
    """
    JSON形式で保存されたタグ付け結果を読み込む
    
    Args:
        file_path: 読み込むファイルパス
        
    Returns:
        Dict[str, Any]: 読み込まれたタグ付け結果
        
    Raises:
        FileNotFoundError: ファイルが存在しない場合
        json.JSONDecodeError: JSONの形式が無効な場合
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"指定されたファイル '{file_path}' が見つかりませんでした。")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSONファイルの形式が無効です: {str(e)}", e.doc, e.pos)