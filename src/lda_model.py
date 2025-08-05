import MeCab
from collections import defaultdict
import gensim
from gensim.models import CoherenceModel

NEOLOGD_DICT_PATH = "/opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd"
MECAB_CONFIG_PATH = "/opt/homebrew/etc/mecabrc"

class LDATopicModel:
    def __init__(self):
        """
        MeCabの設定を行う。
        """
        self.mecab_tagger = MeCab.Tagger(f'-d {NEOLOGD_DICT_PATH} -r {MECAB_CONFIG_PATH}')
        self.lda_model = None   # LDAモデルのプレースホルダー
        self.corpus = None
        self.dictionary = None

    def _parse_text_with_mecab(
            self,
            text: str,
            target_pos: list = ['名詞', '動詞', '形容詞'],      # 抽出したい品詞
            excluded_pos_detail: list = ['数', '非自立', '接尾'],  # 除外したい品詞の詳細項目
            stopwords: list = ['する', 'ある'],              # 除去したいストップワード
            ) -> list:
        """
        テキストを形態素解析し、指定された条件に基づいて単語を抽出する。

        Args:
            text (str): 入力テキスト。
            target_pos (list): 抽出する品詞のリスト。
            excluded_pos_detail (list): 除外する品詞の詳細項目のリスト。
            stopwords (list): 除去する単語のリスト。

        Returns:
            list: 条件を満たす単語のリスト。
        """
        # MeCabで形態素解析
        parsed_result = self.mecab_tagger.parse(text)

        # 解析結果を行単位で分割
        lines = parsed_result.split('\n')

        # 各行をカンマで分割し、詳細項目をリスト化
        tokenized_lines = [line.split(',') for line in lines if line != ""]

        # タブで分割して元の単語と品詞情報を抽出
        tokens = [token[0].split('\t') + token[1:] for token in tokenized_lines if len(token) > 1]

        # 品詞フィルタリング（指定された品詞のみ残す）
        tokens_filtered_by_pos = [token for token in tokens if len(token) > 1 and token[1] in target_pos]

        # 詳細項目フィルタリング（不要な項目を除外）
        tokens_filtered_by_detail = [token for token in tokens_filtered_by_pos if len(token) > 2 and token[2] not in excluded_pos_detail]

        # 原型を抽出し、ストップワードを除去
        base_forms = [token[7] for token in tokens_filtered_by_detail if len(token) > 7 and token[7] not in stopwords]

        return base_forms

    def parse_multiple_texts(
            self,
            texts: list[str],
            target_pos: list = ['名詞', '動詞', '形容詞'],      # 抽出したい品詞
            excluded_pos_detail: list = ['数', '非自立', '接尾'],  # 除外したい品詞の詳細項目
            stopwords: list = ['する', 'ある'],               # 除去したいストップワード
            ) -> list[list[str]]:
        """
        複数のテキストを解析し、単語を抽出する。

        Args:
            texts (list[str]): 入力テキストのリスト。
            target_pos (list): 抽出する品詞のリスト。
            excluded_pos_detail (list): 除外する品詞の詳細項目のリスト。
            stopwords (list): 除去する単語のリスト。

        Returns:
            list[list[str]]: 各テキストから抽出された単語のリスト。
        """
        extracted_words_list = []
        for text in texts:
            extracted_words = self._parse_text_with_mecab(
                text=text,
                target_pos=target_pos,
                excluded_pos_detail=excluded_pos_detail,
                stopwords=stopwords,
            )
            extracted_words_list.append(extracted_words)
        return extracted_words_list

    def build(
            self,
            raw_texts: list[str],
            min_freq: int = 1,
            num_topics: int = 5,
            alpha: float | str = 'auto',
            eta: float | str = 'auto',
            use_best: bool = False
        ) -> None:
        """
        LDAモデルを構築する。

        Args:
            parsed_texts (list[list[str]]): 解析済みテキストのリスト。
            min_freq (int): 単語の最小出現頻度。
            num_topics (int): トピック数。
            alpha (float or str): トピック分布のディリクレパラメータ。
            eta (float or str): 単語分布のディリクレパラメータ。
        """
        #{'num_topics': num, 'alpha': alpha, 'eta': eta}
        if use_best:
            try:
                num_topics = self.best_params["num_topics"]
                alpha = self.best_params["alpha"]
                eta = self.best_params["eta"]
            except AttributeError:
                print("not found best paras, use default")

        word_freq = defaultdict(int)

        parsed_texts = self.parse_multiple_texts(raw_texts)

        # 単語の出現回数をカウント
        for text in parsed_texts:
            for word in text:
                word_freq[word] += 1

        # 最小頻度を超える単語のみを残す
        filtered_texts = [[word for word in text if word_freq[word] > min_freq] for text in parsed_texts]

        self.dictionary = gensim.corpora.Dictionary(filtered_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in filtered_texts]
        self.lda_model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            num_topics=num_topics,
            id2word=self.dictionary,
            alpha=alpha,
            eta=eta,
            passes=10,
            random_state=100,
            update_every=1,
            chunksize=100,
            per_word_topics=True
        )

    def _compute_coherence(self, texts, coherence='c_v') -> float:
        """
        コヒーレンススコアを計算する。

        Args:
            texts (list[list[str]]): トピックモデルのコヒーレンスを評価するテキスト。
            coherence (str): コヒーレンスの種類。

        Returns:
            float: コヒーレンススコア。
        """
        coherence_model = CoherenceModel(model=self.lda_model, texts=texts, dictionary=self.dictionary, coherence=coherence)
        return coherence_model.get_coherence()

    def train(
            self,
            raw_texts: list[str],
            min_freq = 1,
            topic_nums = [x for x in range(2, 21)],
            alphas = [0.01, 0.1, 'symmetric', 'asymmetric'],
            etas = [0.01, 0.1, 'auto']
            ) -> dict:
        """
        グリッドサーチを実行して最適なパラメータでモデルを訓練する。

        Args:
            raw_texts (list[str]): 生のテキストデータのリスト。

        Returns:
            dict: 最適なパラメータとそのコヒーレンススコア。
        """
        extracted = self.parse_multiple_texts(texts=raw_texts)

        best_coherence = -1
        best_params = {}

        for num in topic_nums:
            for alpha in alphas:
                for eta in etas:
                    print(f"トピック数: {num}, alpha: {alpha}, eta: {eta}")
                    self.build(raw_texts, min_freq=min_freq, num_topics=num, alpha=alpha, eta=eta)
                    coherence = self._compute_coherence(extracted)
                    print(f"コヒーレンススコア: {coherence}\n")

                    if coherence > best_coherence:
                        best_coherence = coherence
                        self.best_params = {'num_topics': num, 'alpha': alpha, 'eta': eta}

        print(f"最適なパラメータ: {self.best_params} with コヒーレンススコア: {best_coherence}")
        return {'best_params': best_params, 'best_coherence': best_coherence}

    def show_topics(self, topic_max_len = 10, word_max_len = 10):
        """
        各ドキュメントのトピック分布を表示する。

        Args:
            raw_texts (list[str]): 生のテキストデータのリスト。
        """
        return self.lda_model.show_topics(topic_max_len, word_max_len)
        
        

    def analyze_single_text(self, text: str, len: int = 5) -> list:
        """
        単一のテキストを解析し、上位5つのトピックの単語群を返す。

        Args:
            text (str): 入力テキスト。

        Returns:
            list: 上位5つのトピックに関連する単語群のリスト。
        """
        extracted = self._parse_text_with_mecab(text)
        bow = self.dictionary.doc2bow(extracted)

        # トピック分布を取得
        topic_distribution = self.lda_model.get_document_topics(bow)  # (トピックID, 確率)のリストを取得
        topic_distribution = sorted(topic_distribution, key=lambda x: -x[1])[:len]  # 確率が高い順に上位5つを取得

        #print(topic_distribution)

        # トピックごとの単語を取得
        top_topics = []
        for topic_id, _ in topic_distribution:
            words = [word for word, _ in self.lda_model.show_topic(topic_id, topn=10)]  # トピックの単語を取得
            top_topics.append(words)

        return topic_distribution, top_topics
    
    def analyze_single_text_ad(self, text: str, len: int = 5, word_max_len: int = 10) -> list[dict]:
        """
        単一のテキストを解析し、トピックの割合と関連単語を含むリストを返す。

        Args:
            text (str): 入力テキスト。
            len (int): 取得する上位トピック数（デフォルトは5）。
            word_max_len (int): 各トピックから取得する単語の最大数（デフォルトは10）。

        Returns:
            list[dict]: 各トピックの情報を含む辞書のリスト。
                        - "id": トピックID
                        - "weight": トピックの単一文書における割合
                        - "distribution": トピックに関連する単語リスト
        """
        extracted = self._parse_text_with_mecab(text)  # MeCabで前処理
        bow = self.dictionary.doc2bow(extracted)  # BOWに変換

        # トピック分布を取得 (トピックID, 確率) のリスト
        topic_distribution = self.lda_model.get_document_topics(bow)
        topic_distribution = sorted(topic_distribution, key=lambda x: -x[1])[:len]  # 上位 `len` 個を取得

        # 各トピックの情報を取得
        results = []
        for topic_id, weight in topic_distribution:
            words = [word for word, _ in self.lda_model.show_topic(topic_id, topn=word_max_len)]  # トピックの単語を取得
            results.append({
                "id": topic_id,
                "weight": weight,
                "distribution": words
            })

        return results
    

    def analyze_single_text_ad2(self, text: str, len: int = 5, word_max_len: int = 10) -> list[dict]:
        """
        単一のテキストを解析し、トピックの割合と関連単語を含むリストを返す。
        各トピック内の単語ごとの重みも取得。

        Args:
            text (str): 入力テキスト。
            len (int): 取得する上位トピック数（デフォルトは5）。
            word_max_len (int): 各トピックから取得する単語の最大数（デフォルトは10）。

        Returns:
            list[dict]: 各トピックの情報を含む辞書のリスト。
                        - "id": トピックID
                        - "weight": トピックの単一文書における割合
                        - "distribution": トピックに関連する単語リスト
                            - 各単語は { "word": (単語), "weight": (その単語の重み) } の形式
        """
        extracted = self._parse_text_with_mecab(text)  # MeCabで前処理
        bow = self.dictionary.doc2bow(extracted)  # BOWに変換

        # トピック分布を取得 (トピックID, 確率) のリスト
        topic_distribution = self.lda_model.get_document_topics(bow)
        topic_distribution = sorted(topic_distribution, key=lambda x: -x[1])[:len]  # 上位 `len` 個を取得

        # 各トピックの情報を取得
        results = []
        for topic_id, weight in topic_distribution:
            words_with_weights = [
                {"word": word, "weight": word_weight}
                for word, word_weight in self.lda_model.show_topic(topic_id, topn=word_max_len)
            ]  # トピックの単語とその重みを取得

            results.append({
                "id": topic_id,
                "weight": weight,
                "distribution": words_with_weights
            })

        return results

    def analyze_single_text_ad3(
        self,
        text: str,
        topic_max_len: int = 5,
        word_max_len: int = 10, 
        topic_weight_threshold: float = 0.0,
        word_weight_threshold: float = 0.0
    ) -> list[dict]:
        """
        単一のテキストを解析し、トピックの割合と関連単語を含むリストを返す。
        各トピックとその単語に閾値を設定可能。

        Args:
            text (str): 入力テキスト。
            len (int): 取得する上位トピック数（デフォルトは5）。
            word_max_len (int): 各トピックから取得する単語の最大数（デフォルトは10）。
            topic_weight_threshold (float): 取得するトピックの重みの閾値（デフォルトは0.0）。
            word_weight_threshold (float): 取得する単語の重みの閾値（デフォルトは0.0）。

        Returns:
            list[dict]: 各トピックの情報を含む辞書のリスト。
                        - "id": トピックID
                        - "weight": トピックの単一文書における割合
                        - "distribution": 閾値以上の単語リスト
                            - 各単語は { "word": (単語), "weight": (その単語の重み) } の形式
        """
        extracted = self._parse_text_with_mecab(text)  # MeCabで前処理
        bow = self.dictionary.doc2bow(extracted)  # BOWに変換

        # トピック分布を取得 (トピックID, 確率) のリスト
        topic_distribution = self.lda_model.get_document_topics(bow)
        topic_distribution = sorted(topic_distribution, key=lambda x: -x[1])[:topic_max_len]  # 上位 `len` 個を取得

        # トピックの閾値を適用
        results = []
        for topic_id, weight in topic_distribution:
            if weight < topic_weight_threshold:
                continue  # トピックの重みが閾値以下ならスキップ

            words_with_weights = [
                {"word": word, "weight": word_weight}
                for word, word_weight in sorted(self.lda_model.show_topic(topic_id, topn=word_max_len), key=lambda x: -x[1])
                if word_weight >= word_weight_threshold  # 単語の重みの閾値を適用
            ]

            results.append({
                "id": topic_id,
                "weight": weight,
                "distribution": words_with_weights
            })

        return results





if __name__ == "__main__":
    import csv

    def read_csv_column(file_path, column_name):
        """
        指定されたパスにあるCSVファイルを読み込み、指定された列の全データをリストとして返す。

        Args:
            file_path (str): CSVファイルのパス
            column_name (str): データを取得する列の名前

        Returns:
            list: 指定された列のデータを含むリスト

        Raises:
            FileNotFoundError: 指定されたファイルが存在しない場合
            KeyError: 指定された列名がCSVに存在しない場合
        """
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                if column_name not in reader.fieldnames:
                    raise KeyError(f"指定された列名 '{column_name}' がCSVファイルに存在しません。")

                return [row[column_name] for row in reader]
        except FileNotFoundError:
            raise FileNotFoundError(f"指定されたファイル '{file_path}' が見つかりませんでした。")
        except Exception as e:
            raise e

    documents = read_csv_column("/Users/taketake/2nd_q4/deim/script/data/livedoor/it-life-hack.csv","body")
    model = LDATopicModel()
    #model.train(documents)
    model.build(documents, 1, 8, 0.1, 0.01)

    #print(documents[10])
    result = model.analyze_single_text(documents[473])
    print(result[0], result[1])