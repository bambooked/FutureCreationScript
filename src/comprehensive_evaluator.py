"""
教師なしタグ付けモデルの総合評価フレームワーク

new_evaluter_idea.mdに基づいた9指標による多面的評価システム。
3つの視点（文書-タグ整合性、タグ検索性能、タグ集合健全性）から
各3つの指標を用いて、タグ付けモデルの性能を総合的に評価します。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.metrics import ndcg_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_mutual_info_score
from sentence_transformers import SentenceTransformer
import umap
from collections import Counter
import math
import json
from abc import ABC, abstractmethod


@dataclass
class EvaluationResult:
    """評価結果を格納するデータクラス"""
    # 文書-タグ整合性（Micro）
    sbert_cosine: float = 0.0
    npmi: float = 0.0
    topic_coherence_cv: float = 0.0
    
    # タグ検索性能（Macro）
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    f1_tag_classification: float = 0.0
    
    # タグ集合健全性
    gini_coefficient: float = 0.0
    avg_tag_cosine_similarity: float = 0.0
    umap_ami_score: float = 0.0
    
    # メタ情報
    method_name: str = ""
    num_documents: int = 0
    num_unique_tags: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """結果を辞書形式に変換"""
        return {
            "sbert_cosine": self.sbert_cosine,
            "npmi": self.npmi,
            "topic_coherence_cv": self.topic_coherence_cv,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "f1_tag_classification": self.f1_tag_classification,
            "gini_coefficient": self.gini_coefficient,
            "avg_tag_cosine_similarity": self.avg_tag_cosine_similarity,
            "umap_ami_score": self.umap_ami_score,
            "method_name": self.method_name,
            "num_documents": self.num_documents,
            "num_unique_tags": self.num_unique_tags
        }


class ComprehensiveEvaluator:
    """
    タグ付けモデルの総合評価クラス
    
    9つの指標を用いて、タグ付けの品質を多面的に評価します。
    """
    
    def __init__(self, 
                 sbert_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 ndcg_k: int = 10):
        """
        Args:
            sbert_model_name: Sentence-BERTモデル名
            ndcg_k: nDCG@kのk値
        """
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.ndcg_k = ndcg_k
    
    def evaluate(self, 
                 documents: List[str], 
                 tags_dict: Dict[int, Dict[int, str]],
                 method_name: str = "Unknown") -> EvaluationResult:
        """
        タグ付け結果を総合的に評価
        
        Args:
            documents: 文書リスト
            tags_dict: {文書ID: {トピックID: タグ}} の辞書
            method_name: 評価対象の手法名
            
        Returns:
            EvaluationResult: 9指標の評価結果
        """
        result = EvaluationResult(method_name=method_name, num_documents=len(documents))
        
        # タグ情報の整理
        all_tags = []
        doc_tag_pairs = []
        for doc_id, doc_tags in tags_dict.items():
            if doc_id < len(documents):
                for topic_id, tag in doc_tags.items():
                    all_tags.append(tag)
                    doc_tag_pairs.append((documents[doc_id], list(doc_tags.values())))
        
        result.num_unique_tags = len(set(all_tags))
        
        # 1. 文書-タグ整合性（Micro）
        result.sbert_cosine = self._calculate_sbert_cosine(doc_tag_pairs)
        result.npmi = self._calculate_npmi(doc_tag_pairs)
        result.topic_coherence_cv = self._calculate_topic_coherence(doc_tag_pairs)
        
        # 2. タグ検索性能（Macro）
        result.ndcg_at_k = self._calculate_ndcg(documents, tags_dict)
        result.mrr = self._calculate_mrr(documents, tags_dict)
        result.f1_tag_classification = self._calculate_f1_classification(documents, tags_dict)
        
        # 3. タグ集合健全性
        result.gini_coefficient = self._calculate_gini(all_tags)
        result.avg_tag_cosine_similarity = self._calculate_avg_tag_similarity(list(set(all_tags)))
        result.umap_ami_score = self._calculate_umap_ami(documents, tags_dict)
        
        return result
    
    def _calculate_sbert_cosine(self, doc_tag_pairs: List[Tuple[str, List[str]]]) -> float:
        """文書とタグのSBERTコサイン類似度を計算"""
        if not doc_tag_pairs:
            return 0.0
        
        similarities = []
        for doc_text, tags in doc_tag_pairs:
            if not tags:
                continue
            
            # 文書とタグをエンコード
            doc_embedding = self.sbert_model.encode([doc_text], normalize_embeddings=True)
            tag_embeddings = self.sbert_model.encode(tags, normalize_embeddings=True)
            
            # コサイン類似度を計算
            sim = cosine_similarity(doc_embedding, tag_embeddings)
            similarities.append(np.mean(sim))
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_npmi(self, doc_tag_pairs: List[Tuple[str, List[str]]]) -> float:
        """正規化点別相互情報量（NPMI）を計算"""
        if not doc_tag_pairs:
            return 0.0
        
        # 簡略版：タグが文書中に出現する確率ベースで計算
        npmi_scores = []
        
        for doc_text, tags in doc_tag_pairs:
            if not tags:
                continue
            
            doc_words = set(doc_text.lower().split())
            for tag in tags:
                tag_words = set(tag.lower().split())
                
                # 共起確率の簡易計算
                overlap = len(doc_words & tag_words)
                if overlap > 0:
                    p_joint = overlap / len(doc_words)
                    p_tag = len(tag_words) / (len(doc_words) + len(tag_words))
                    p_doc = 1.0  # 文書は常に存在
                    
                    pmi = math.log(p_joint / (p_tag * p_doc)) if p_tag > 0 else 0
                    npmi = pmi / (-math.log(p_joint)) if p_joint > 0 else 0
                    npmi_scores.append(npmi)
        
        return np.mean(npmi_scores) if npmi_scores else 0.0
    
    def _calculate_topic_coherence(self, doc_tag_pairs: List[Tuple[str, List[str]]]) -> float:
        """トピック一貫性C_vを計算（簡略版）"""
        if not doc_tag_pairs:
            return 0.0
        
        # タグ集合の語彙的まとまりを評価
        coherence_scores = []
        
        for _, tags in doc_tag_pairs:
            if len(tags) < 2:
                continue
            
            # タグ間の類似度を計算
            tag_embeddings = self.sbert_model.encode(tags, normalize_embeddings=True)
            sim_matrix = cosine_similarity(tag_embeddings)
            
            # 対角線を除いた平均類似度
            mask = np.ones_like(sim_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_sim = np.mean(sim_matrix[mask])
            
            coherence_scores.append(avg_sim)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_ndcg(self, documents: List[str], tags_dict: Dict[int, Dict[int, str]]) -> float:
        """nDCG@kを計算"""
        if not tags_dict:
            return 0.0
        
        # タグベースの文書検索をシミュレート
        unique_tags = set()
        for doc_tags in tags_dict.values():
            unique_tags.update(doc_tags.values())
        
        if not unique_tags:
            return 0.0
        
        ndcg_scores = []
        
        # 各タグに対して検索性能を評価
        for target_tag in list(unique_tags)[:20]:  # 計算量を抑えるため上位20タグ
            # 真のrelevance（タグを持つ文書は1、それ以外は0）
            true_relevance = []
            predicted_scores = []
            
            for doc_id, doc_text in enumerate(documents):
                if doc_id in tags_dict and target_tag in tags_dict[doc_id].values():
                    true_relevance.append(1)
                else:
                    true_relevance.append(0)
                
                # 予測スコア（文書とタグの類似度）
                doc_emb = self.sbert_model.encode([doc_text], normalize_embeddings=True)
                tag_emb = self.sbert_model.encode([target_tag], normalize_embeddings=True)
                score = cosine_similarity(doc_emb, tag_emb)[0][0]
                predicted_scores.append(score)
            
            if sum(true_relevance) > 0:  # 少なくとも1つの関連文書がある場合
                ndcg = ndcg_score([true_relevance], [predicted_scores], k=self.ndcg_k)
                ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def _calculate_mrr(self, documents: List[str], tags_dict: Dict[int, Dict[int, str]]) -> float:
        """Mean Reciprocal Rank (MRR)を計算"""
        if not tags_dict:
            return 0.0
        
        unique_tags = set()
        for doc_tags in tags_dict.values():
            unique_tags.update(doc_tags.values())
        
        if not unique_tags:
            return 0.0
        
        reciprocal_ranks = []
        
        for target_tag in list(unique_tags)[:20]:
            # 文書とタグの類似度でランキング
            scores = []
            relevant_docs = []
            
            for doc_id, doc_text in enumerate(documents):
                doc_emb = self.sbert_model.encode([doc_text], normalize_embeddings=True)
                tag_emb = self.sbert_model.encode([target_tag], normalize_embeddings=True)
                score = cosine_similarity(doc_emb, tag_emb)[0][0]
                scores.append((doc_id, score))
                
                if doc_id in tags_dict and target_tag in tags_dict[doc_id].values():
                    relevant_docs.append(doc_id)
            
            if relevant_docs:
                # スコア順にソート
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # 最初の関連文書の順位を探す
                for rank, (doc_id, _) in enumerate(scores, 1):
                    if doc_id in relevant_docs:
                        reciprocal_ranks.append(1.0 / rank)
                        break
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _calculate_f1_classification(self, documents: List[str], tags_dict: Dict[int, Dict[int, str]]) -> float:
        """タグをone-hot特徴として疑似分類タスクのF1スコアを計算"""
        if not tags_dict or len(documents) < 10:
            return 0.0
        
        # データを訓練/テストに分割
        split_idx = int(len(documents) * 0.7)
        
        # タグをone-hot化
        all_tags = set()
        for doc_tags in tags_dict.values():
            all_tags.update(doc_tags.values())
        
        tag_to_idx = {tag: i for i, tag in enumerate(sorted(all_tags))}
        
        # 特徴量とラベルを作成（簡易的な疑似タスク）
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        for doc_id in range(len(documents)):
            if doc_id not in tags_dict:
                continue
            
            # one-hot特徴量
            features = [0] * len(tag_to_idx)
            for tag in tags_dict[doc_id].values():
                if tag in tag_to_idx:
                    features[tag_to_idx[tag]] = 1
            
            # 疑似ラベル（最初のタグのインデックス）
            first_tag = list(tags_dict[doc_id].values())[0] if tags_dict[doc_id] else None
            if first_tag and first_tag in tag_to_idx:
                label = tag_to_idx[first_tag]
                
                if doc_id < split_idx:
                    X_train.append(features)
                    y_train.append(label)
                else:
                    X_test.append(features)
                    y_test.append(label)
        
        if len(X_train) < 5 or len(X_test) < 5:
            return 0.0
        
        # 単純な最近傍分類器で予測
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        return f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    def _calculate_gini(self, all_tags: List[str]) -> float:
        """タグ分布のGini係数を計算"""
        if not all_tags:
            return 0.0
        
        # タグの出現頻度をカウント
        tag_counts = Counter(all_tags)
        counts = sorted(tag_counts.values())
        
        n = len(counts)
        cumsum = 0
        for i, count in enumerate(counts):
            cumsum += count * (n - i)
        
        return (n + 1 - 2 * cumsum / sum(counts)) / n
    
    def _calculate_avg_tag_similarity(self, unique_tags: List[str]) -> float:
        """タグ間の平均コサイン類似度を計算"""
        if len(unique_tags) < 2:
            return 0.0
        
        # すべてのタグをエンコード
        tag_embeddings = self.sbert_model.encode(unique_tags, normalize_embeddings=True)
        
        # 類似度行列を計算
        sim_matrix = cosine_similarity(tag_embeddings)
        
        # 対角線を除いた平均を計算
        mask = np.ones_like(sim_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        return np.mean(sim_matrix[mask])
    
    def _calculate_umap_ami(self, documents: List[str], tags_dict: Dict[int, Dict[int, str]]) -> float:
        """UMAP + AMIによるタグと文書クラスタの対応度を計算"""
        if len(documents) < 10:
            return 0.0
        
        # 文書のエンベディングを取得
        doc_embeddings = self.sbert_model.encode(documents, normalize_embeddings=True)
        
        # UMAPで次元削減
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(documents)-1))
        doc_embeddings_2d = reducer.fit_transform(doc_embeddings)
        
        # クラスタリング（簡易版：k-means）
        from sklearn.cluster import KMeans
        n_clusters = min(5, len(documents) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(doc_embeddings_2d)
        
        # タグラベルの作成（最頻出タグを文書のラベルとする）
        tag_labels = []
        for doc_id in range(len(documents)):
            if doc_id in tags_dict and tags_dict[doc_id]:
                # 最初のタグを使用
                first_tag = list(tags_dict[doc_id].values())[0]
                tag_labels.append(first_tag)
            else:
                tag_labels.append("no_tag")
        
        # ラベルを数値に変換
        unique_labels = list(set(tag_labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        tag_labels_numeric = [label_to_idx[label] for label in tag_labels]
        
        # AMIスコアを計算
        return adjusted_mutual_info_score(cluster_labels, tag_labels_numeric)


class EvaluationPipeline:
    """
    評価パイプライン全体を管理するクラス
    """
    
    def __init__(self, evaluator: ComprehensiveEvaluator):
        """
        Args:
            evaluator: 評価器インスタンス
        """
        self.evaluator = evaluator
        self.results = {}
    
    def evaluate_multiple_methods(self, 
                                  documents: List[str],
                                  methods_tags: Dict[str, Dict[int, Dict[int, str]]]) -> pd.DataFrame:
        """
        複数の手法を評価して比較
        
        Args:
            documents: 文書リスト
            methods_tags: {手法名: {文書ID: {トピックID: タグ}}} の辞書
            
        Returns:
            pd.DataFrame: 評価結果の比較表
        """
        for method_name, tags_dict in methods_tags.items():
            result = self.evaluator.evaluate(documents, tags_dict, method_name)
            self.results[method_name] = result
        
        # DataFrameに変換
        df_data = []
        for method_name, result in self.results.items():
            df_data.append(result.to_dict())
        
        return pd.DataFrame(df_data)
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """
        評価レポートを生成
        
        Args:
            output_path: 出力ファイルパス
        """
        report = {
            "evaluation_results": {},
            "summary": {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 各手法の結果を整理
        for method_name, result in self.results.items():
            report["evaluation_results"][method_name] = result.to_dict()
        
        # サマリー統計
        if self.results:
            # 各指標の平均値
            metrics = ["sbert_cosine", "npmi", "topic_coherence_cv", 
                      "ndcg_at_k", "mrr", "f1_tag_classification",
                      "gini_coefficient", "avg_tag_cosine_similarity", "umap_ami_score"]
            
            for metric in metrics:
                values = [getattr(result, metric) for result in self.results.values()]
                report["summary"][f"{metric}_mean"] = np.mean(values)
                report["summary"][f"{metric}_std"] = np.std(values)
        
        # ファイルに保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report


# 使用例
if __name__ == "__main__":
    print("教師なしタグ付けモデルの総合評価フレームワーク")
    print("=" * 60)
    print("このモジュールは以下の9指標で評価を行います：")
    print("\n【文書-タグ整合性（Micro）】")
    print("1. SBERT Cosine: 文書とタグの意味的類似度")
    print("2. NPMI: 正規化点別相互情報量")
    print("3. Topic Coherence C_v: タグ集合の語彙的まとまり")
    print("\n【タグ検索性能（Macro）】")
    print("4. nDCG@k: タグ→文書検索のランキング品質")
    print("5. MRR: 最初の関連文書出現順位")
    print("6. F1 (Tag Classification): タグone-hotによる疑似分類性能")
    print("\n【タグ集合健全性】")
    print("7. Gini係数: タグ人気度の偏り")
    print("8. 平均タグ間コサイン: タグの冗長性")
    print("9. UMAP + AMI: タグと文書クラスタの対応度")
    print("\n使用方法:")
    print("evaluator = ComprehensiveEvaluator()")
    print("result = evaluator.evaluate(documents, tags_dict)")
    