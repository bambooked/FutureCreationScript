"""
データローダー

CSVファイルから指定された列のデータを読み込む機能
script2のcsv_loader.pyを参考に実装
"""
import csv
from typing import List


def read_csv_column(file_path: str, column_name: str) -> List[str]:
    """
    指定されたパスにあるCSVファイルを読み込み、指定された列の全データをリストとして返す。

    Args:
        file_path (str): CSVファイルのパス
        column_name (str): データを取得する列の名前

    Returns:
        List[str]: 指定された列のデータを含むリスト

    Raises:
        FileNotFoundError: 指定されたファイルが存在しない場合
        KeyError: 指定された列名がCSVに存在しない場合
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # 列名の存在確認
            if column_name not in reader.fieldnames:
                raise KeyError(f"指定された列名 '{column_name}' がCSVファイルに存在しません。")

            # データを読み込んでリストに変換
            column_data = [row[column_name] for row in reader]
            return column_data
            
    except FileNotFoundError:
        raise FileNotFoundError(f"指定されたファイル '{file_path}' が見つかりませんでした。")
    except Exception as e:
        # その他の例外は再発生させる
        raise e