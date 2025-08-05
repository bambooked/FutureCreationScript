# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

fc-scriptは、教師なしタグ付けモデルの評価フレームワークを提供するPythonプロジェクトです。主な機能は以下の通りです：

- LDAトピックモデリング（`src/lda_model.py`）
- 包括的な評価指標（`src/comprehensive_evaluator.py`）
- LLMクライアント（`src/llm_client.py`）

## 開発環境のセットアップと実行

### 依存関係の管理
このプロジェクトはuvを使用して依存関係を管理しています。

```bash
# 仮想環境の作成と依存関係のインストール
uv venv
uv pip sync

# メインスクリプトの実行
uv run python main.py

# 特定のモジュールを直接実行
uv run python src/lda_model.py
uv run python src/comprehensive_evaluator.py
```

### 主要な依存関係
- gensim (LDAモデリング)
- mecab (日本語形態素解析)
- scikit-learn (機械学習ユーティリティ)
- sentence-transformers (文書埋め込み)
- pandas, numpy (データ処理)
- umap (次元削減)

## コード実行時の重要な環境変数

```bash
# LLMClient使用時に必要
export OPENROUTER_API_KEY="your-api-key"
```

## アーキテクチャ概要

このプロジェクトは教師なしタグ付けモデルの評価を行うためのフレームワークで、3つの主要なコンポーネントから構成されています。各コンポーネントは独立して動作可能で、組み合わせて使用することで包括的な評価パイプラインを構築できます。

### 1. LDATopicModel (`src/lda_model.py`)
- MeCabを使用した日本語テキストの前処理
- LDAモデルの構築と学習
- グリッドサーチによるハイパーパラメータ最適化
- 単一文書のトピック分析機能

**重要な設定:**
- MeCab辞書パス: `/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd`
- MeCab設定パス: `/opt/homebrew/etc/mecabrc`

### 2. ComprehensiveEvaluator (`src/comprehensive_evaluator.py`)
タグ付けモデルを9つの指標で評価：

**文書-タグ整合性（Micro）:**
- SBERT Cosine類似度
- NPMI（正規化点別相互情報量）
- Topic Coherence C_v

**タグ検索性能（Macro）:**
- nDCG@k
- MRR（Mean Reciprocal Rank）
- F1スコア（タグ分類）

**タグ集合健全性:**
- Gini係数
- 平均タグ間コサイン類似度
- UMAP + AMI

### 3. LLMClient (`src/llm_client.py`)
- OpenRouter APIを使用したLLMとの通信
- 環境変数`OPENROUTER_API_KEY`が必要
- モデル切り替え機能（デフォルト: gpt-3.5-turbo）

## 主要なデータフロー

1. **文書の前処理**: `LDATopicModel`が日本語文書をMeCabで形態素解析し、品詞フィルタリングとストップワード除去を行う
2. **トピックモデリング**: gensimのLDAを使用してトピックを抽出し、各文書に対してトピック分布を計算
3. **タグ生成**: 各トピックから代表的な単語を抽出してタグとして使用
4. **評価**: `ComprehensiveEvaluator`が生成されたタグを9つの観点から評価し、モデルの品質を定量化

## 使用例

### LDAモデルの学習と使用
```python
from src.lda_model import LDATopicModel

# モデルの初期化と学習
model = LDATopicModel()
model.build(documents, min_freq=1, num_topics=8, alpha=0.1, eta=0.01)

# 単一文書の分析
topic_dist, top_topics = model.analyze_single_text("分析したいテキスト")

# グリッドサーチによる最適化
best_params = model.train(documents)
model.build(documents, use_best=True)  # 最適パラメータで再構築
```

### 評価の実行
```python
from src.comprehensive_evaluator import ComprehensiveEvaluator, EvaluationPipeline

# 単一手法の評価
evaluator = ComprehensiveEvaluator()
result = evaluator.evaluate(documents, tags_dict, method_name="LDA")

# 複数手法の比較
pipeline = EvaluationPipeline(evaluator)
methods_tags = {
    "LDA": lda_tags_dict,
    "Other_Method": other_tags_dict
}
comparison_df = pipeline.evaluate_multiple_methods(documents, methods_tags)
pipeline.generate_report("evaluation_report.json")
```

## 注意事項

- 日本語処理にはMeCabが必要です
- LLMClientを使用する場合は、OPENROUTER_API_KEYの設定が必要です
- データファイルは`data/`ディレクトリに配置してください