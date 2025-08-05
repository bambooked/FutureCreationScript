"""
メイン統合処理

LDAモデルとLLMを組み合わせた文書自動タグ付けシステムのメイン処理
"""
import argparse
from typing import Dict, Any, List
from pathlib import Path

from .lda_model import LDATopicModel  
from .async_llm_client import AsyncLLMClient
from .async_composer import AsyncTaggingComposer
from .comprehensive_evaluator import ComprehensiveEvaluator
from .data_loader import read_csv_column
from .json_handler import save_tagging_results
from .evaluation_converter import convert_to_evaluation_format


async def run_tagging_pipeline(
    csv_file_path: str,
    column_name: str,
    output_path: str,
    lda_params: Dict[str, Any] = None,
    llm_model: str = "google/gemini-2.0-flash-001",
    max_documents: int = None,
    batch_size: int = 10,
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """
    文書自動タグ付けパイプラインを非同期で実行する
    
    Args:
        csv_file_path: 入力CSVファイルのパス
        column_name: 処理する列名（通常は"body"）
        output_path: 結果を保存するJSONファイルのパス
        lda_params: LDAモデルのパラメータ
        llm_model: 使用するLLMモデル名（デフォルト: google/gemini-2.0-flash-001）
        max_documents: 処理する最大文書数（None = 全て）
        batch_size: 並行処理のバッチサイズ
        max_concurrent: 最大同時接続数
        
    Returns:
        Dict[str, Any]: 処理結果のサマリー
    """
    if lda_params is None:
        lda_params = {
            "num_topics": 8,
            "alpha": 0.1,
            "eta": 0.01,
            "min_freq": 1
        }
    
    print("文書自動タグ付けパイプラインを開始します...")
    
    # 1. データ読み込み
    print(f"CSVファイルを読み込み中: {csv_file_path}")
    documents = read_csv_column(csv_file_path, column_name)
    
    if max_documents and len(documents) > max_documents:
        documents = documents[:max_documents]
        print(f"処理文書数を{max_documents}件に制限しました")
    
    print(f"読み込み完了: {len(documents)}件の文書")
    
    # 2. LDAモデルの構築
    print("LDAモデルを構築中...")
    lda_model = LDATopicModel()
    lda_model.build(
        raw_texts=documents,
        min_freq=lda_params.get("min_freq", 1),
        num_topics=lda_params.get("num_topics", 8),
        alpha=lda_params.get("alpha", 0.1),
        eta=lda_params.get("eta", 0.01)
    )
    print("LDAモデル構築完了")
    
    # 3. 非同期LLMクライアントの初期化
    print("非同期LLMクライアントを初期化中...")
    async_llm_client = AsyncLLMClient(max_concurrent=max_concurrent)
    async_llm_client.set_model(llm_model)
    print(f"LLMモデル設定完了: {llm_model}")
    
    # 4. 非同期タグ生成コンポーザーの初期化
    async_composer = AsyncTaggingComposer(lda_model, async_llm_client)
    
    # 5. 並行してタグを生成
    print("文書にタグを並行生成中...")
    
    try:
        # 並行してタグ生成
        tags_list = await async_composer.generate_multiple_tags(
            documents, 
            batch_size=batch_size,
            rate_limit_delay=0.1  # レート制限を考慮
        )
        
        # データを構造化
        documents_data = []
        for doc_id, (document, tags) in enumerate(zip(documents, tags_list)):
            # トピック分析（LDA部分は同期処理）
            try:
                topic_dist = lda_model.analyze_single_text_ad3(
                    document,
                    topic_max_len=5,
                    word_max_len=10,
                    topic_weight_threshold=0.0,
                    word_weight_threshold=0.0
                )
            except Exception as e:
                print(f"\n警告: 文書{doc_id}のトピック分析中にエラー: {str(e)}")
                topic_dist = []
            
            # データを構造化
            doc_data = {
                "doc_id": doc_id,
                "body": document,
                "topics": topic_dist,
                "tags": tags if tags else []  # エラー時は空リスト
            }
            documents_data.append(doc_data)
            
        print(f"\n並行タグ生成完了: {len(documents_data)}件")
        
    except Exception as e:
        print(f"\nエラー: 並行タグ生成中に問題が発生しました: {str(e)}")
        # フォールバック: 空のタグで継続
        documents_data = []
        for doc_id, document in enumerate(documents):
            doc_data = {
                "doc_id": doc_id,
                "body": document,
                "topics": [],
                "tags": []
            }
            documents_data.append(doc_data)
    
    # 6. 結果をJSONで保存
    print(f"結果をJSON形式で保存中: {output_path}")
    metadata = {
        "lda_params": lda_params,
        "llm_model": llm_model,
        "total_documents": len(documents_data)
    }
    save_tagging_results(documents_data, metadata, output_path)
    print("JSON保存完了")
    
    # 7. 評価の実行
    print("タグ付け結果を評価中...")
    try:
        eval_documents, eval_tags_dict = convert_to_evaluation_format(documents_data)
        evaluator = ComprehensiveEvaluator()
        evaluation_result = evaluator.evaluate(
            documents=eval_documents,
            tags_dict=eval_tags_dict,
            method_name="LDA-LLM"
        )
        print("評価完了")
        
        # 評価結果を表示
        eval_dict = evaluation_result.to_dict()
        print("\n=== 評価結果 ===")
        print(f"SBERT Cosine類似度: {eval_dict['sbert_cosine']:.3f}")
        print(f"NPMI: {eval_dict['npmi']:.3f}")
        print(f"Topic Coherence C_v: {eval_dict['topic_coherence_cv']:.3f}")
        print(f"nDCG@k: {eval_dict['ndcg_at_k']:.3f}")
        print(f"MRR: {eval_dict['mrr']:.3f}")
        print(f"F1 Classification: {eval_dict['f1_tag_classification']:.3f}")
        print(f"Gini係数: {eval_dict['gini_coefficient']:.3f}")
        print(f"平均タグ類似度: {eval_dict['avg_tag_cosine_similarity']:.3f}")
        print(f"UMAP AMI: {eval_dict['umap_ami_score']:.3f}")
        
    except Exception as e:
        print(f"評価中にエラーが発生しました: {str(e)}")
        evaluation_result = None
    
    # 8. 結果のサマリーを返す
    result_summary = {
        "documents_processed": len(documents_data),
        "output_file": output_path,
        "lda_params": lda_params,
        "llm_model": llm_model,
        "evaluation_result": evaluation_result.to_dict() if evaluation_result else None
    }
    
    print(f"\nパイプライン完了: {len(documents_data)}件の文書を処理しました")
    return result_summary


async def async_main():
    """非同期メイン関数"""
    parser = argparse.ArgumentParser(
        description="LDAとLLMを組み合わせた文書自動タグ付けシステム"
    )
    
    parser.add_argument(
        "csv_file", 
        help="入力CSVファイルのパス"
    )
    parser.add_argument(
        "--column", 
        default="body",
        help="処理する列名 (デフォルト: body)"
    )
    parser.add_argument(
        "--output", 
        default="tagging_results.json",
        help="出力JSONファイルのパス (デフォルト: tagging_results.json)"
    )
    parser.add_argument(
        "--topics", 
        type=int, 
        default=8,
        help="LDAのトピック数 (デフォルト: 8)"
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.1,
        help="LDAのalphaパラメータ (デフォルト: 0.1)"
    )
    parser.add_argument(
        "--eta", 
        type=float, 
        default=0.01,
        help="LDAのetaパラメータ (デフォルト: 0.01)"
    )
    parser.add_argument(
        "--model", 
        default="google/gemini-2.0-flash-001",
        help="LLMモデル名 (デフォルト: google/gemini-2.0-flash-001)"
    )
    parser.add_argument(
        "--max-docs", 
        type=int,
        help="処理する最大文書数 (指定しない場合は全て処理)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="並行処理のバッチサイズ (デフォルト: 10)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="最大同時接続数 (デフォルト: 10)"
    )
    
    args = parser.parse_args()
    
    # パラメータの設定
    lda_params = {
        "num_topics": args.topics,
        "alpha": args.alpha,
        "eta": args.eta,
        "min_freq": 1
    }
    
    try:
        # 非同期パイプラインの実行
        result = await run_tagging_pipeline(
            csv_file_path=args.csv_file,
            column_name=args.column,
            output_path=args.output,
            lda_params=lda_params,
            llm_model=args.model,
            max_documents=args.max_docs,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent
        )
        
        print(f"\n処理が正常に完了しました。結果は {args.output} に保存されました。")
        return 0
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        return 1


def main():
    """同期版メイン関数（asyncio.run()でラップ）"""
    import asyncio
    return asyncio.run(async_main())


if __name__ == "__main__":
    exit(main())