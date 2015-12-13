# Groonga word2vec plugin
Groongaにword2vecのコマンド、関数を追加します。

* ```dump_to_train_file```  
* ```word2vec_train```  
* ```word2vec_distance```  
* ```word2vec_load```  
* ```word2vec_unload```  
* ```QueryExpanderWord2vec```

## コマンド
### ```dump_to_train_file```
Groongaのカラムに格納されたテキストからword2vec向けの学習用のファイルを生成します。

カラムからの出力に対してMeCabで分かち書きをし、Groongaのノーマライザ―を使って文字の正規化、およびRE2ライブラリを使って正規表現フィルタをかけることができます。

学習用ファイル名が省略された場合、分かち書きされたテキストファイルが`{Groongaのデータベースパス}+_w2v.txt`に出力されます。

| arg        | description | default      |
|:-----------|:------------|:-------------|
| table      | Groongaのテーブル | NULL |
| column      | Groongaのカラム名  ``,``区切りで複数指定可  末尾が``_``の場合、スペースを``_``に置換してフレーズ化する  末尾が``/``の場合、形態素解析する  末尾が``*[2-9]``の場合、その回数だけ繰り返し出力する  末尾が``$``の場合、``<.*>``と``0-9,.;:&^/-−#'"()[]、。【】「」~・``を削除  末尾が``@``の場合、アルファベット削除  末尾が``[.*]``の場合、[]で囲まれたラベルを先頭に追加   カラムの上限数20 | NULL | 
| filter      | Groongaの[スクリプト構文](http://groonga.org/ja/docs/reference/grn_expr/script_syntax.html)で絞り込む | NULL |
| train_file     | 学習用テキストファイル(一時ファイル) |  `{groonga_db}_w2v.txt` |
| normalizer      | Groongaのノーマライザ― NONEの場合なし | NormalizerAuto |
| input_filter   | 入力テキストから除去したい文字列の正規表現(全置換) | NULL |
| mecab_option   | MeCabのオプション Mecab使わない場合NONE | -Owakati |
| sentence_vectors   | sentence vectorを含める場合は1  doc_id:_id(Groongaのtableの_id)の形式で文書ベクトルを追加  (それ以外の単語ベクトルもある) | 0 |

オプションは、通常のGroongaのコマンドと同様に上記の順番で入力する場合は省略することができます。  
上記の順番以外で入力したい場合や省略したい場合は、``--``を先頭につけます。  

* 出力形式  
JSON

* 実行例

```
> dump_to_train_file Entries title,tags
[[0,0.0,0.0],true]
```

### ```word2vec_train```
スペース区切りの学習用テキストからword2vecコマンドで学習させます。

OSが実行可能なパスにword2vecコマンドが必要です。デフォルトでは一緒にbindirにword2vecコマンドがインストールされます。

学習用ファイル名が省略された場合、`{Groongaのデータベースパス}+_w2v.txt`が利用されます。

学習済みモデルファイル名が省略された場合、学習済みモデルファイルが`{Groongaのデータベースパス}+_w2v.bin`に出力されます。

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| train_file     | 学習用テキストファイル(一時ファイル) |  `{groonga_db}_w2v.txt` |
| output_file   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| save_vovab_file    | save_vocab_file | NULL |
| read_vovab_file    | read_vocab_file | NULL |
| threads    | 学習時のスレッド数 | 12 |
| size     |  次元数 | 100 |
| debug    | debug Groongaのログに出力される | 2 |
| binary    | モデルファイルをテキスト形式にする場合は0 | 1 |
| cbow    | skip-gramを使う場合は0  cbowを使う場合は1 | 0 |
| alpha    | alpha | 0.025 cbowの場合0.05 |
| window    | 文脈とする前後の幅 | 5 |
| sample    |  高頻度の単語を無視する閾値 | 1e-3 | 
| hs    | 階層的Softmax(高速化手法,ランダム要素有) | 1 |
| negative    | ネガティブサンプリングの単語数(高速化手法,ランダム要素有) | 0 |
| iter    | 学習回数 | 5 |
| min_count    | 単語の最低出現数  sentence_vectorsの場合は自動的に1になる | 5 |
| classes    | K-meansクラスタリングする場合は1以上の分類したい数 | 0 |
| sentence_vectors   | sentence vectorを含める場合は1  doc_id:_id(Groongaのtableの_id)の形式で文書ベクトルを追加  (それ以外の単語ベクトルもある) | 0 |

* 出力形式  
JSON

* 実行例

```
> word2vec_train
[[0,0.0,0.0],true]
```

### ```word2vec_distance```

入力した単語とベクトル距離が近い単語が出力されます。類似語らしきものを取得することができます。  
"単語A + 単語B"など、スペースと+-で単語の足し引きをさせることができます。
出力結果の件数制限、オフセット、閾値をサポートしています。 
学習済みモデルファイルがロードされていない場合、自動的にロードされます。 
ロードされていない場合にファイル名が省略されると、`{Groongaのデータベースパス}+_w2v.bin`がロードされます。  

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| term      | 入力単語 or 単語式 (e.g. 単語A + 単語B - 単語C) | NULL |
| offset      | 結果出力のオフセット | 0 | 
| limit     | 結果出力の上限件数 n_sort以上の数は出力されない | 10 |
| n_sort     | 挿入ソートのバッファサイズ 増やせば増やすほど遅くなる | 40 |
| threshold     | コサイン距離(_value)の閾値、1以下の小数を指定 | -1 |
| normalizer      | Groongaのノーマライザ― | NormalizerAuto |
| prefix_filter   | 出力をさせる単語に前方一致する文字列  高速な絞込が可能 | NULL |
| stop_filter   | 出力をさせない単語にマッチする正規表現(完全一致) | NULL |
| output_filter   | 出力をさせる単語から除去する正規表現(全置換) | NULL |
| mecab_option   | MeCabのオプション | NULL |
| file_path   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| binary    | テキスト形式のモデルファイルを使う場合は0 | 1 |
| is_phrase   | スペースを``_``に置換してフレーズ化する場合1 | 0 |
| edit_distance   | 出力結果を編集距離の近い順にする場合1  編集距離は_scoreにセットされる | 0 |
| pca   | PCA(主成分分析)をして削減する次元数   可視化などの利用を想定  1以上の場合コサイン距離と次元圧縮したベクトルが出力される  入力値のベクトルも出力される | 0 |
| pca_centered   | PCA(主成分分析)をするときにセンタリングさせる | 1 |
| expander_mode   | 出力形式をクエリ展開用にするかどうかのフラグ  1:クエリ展開 ((query1) OR (query2)) | 0 |
| sentence_vectors   | doc_id:から始まるsentence_vectorのみを出力する場合1  これを使う場合prefix_filterは無視される | 0 |
| table   | sentence_vectorのdoc_idに対応させるテーブル名 | NULL |
| column   | sentence_vectorのdoc_idに対応して出力するカラム名  ``,``区切りで複数指定可  _scoreはfloat出力できないため0と出力される(ソートはされている) | _id,_score |
| sortby   | sentence_vectorのdoc_idに対応して出力するカラムのソート  ``,``区切りで複数指定可 | -_score |

* 上限

語彙の最大バイト数(max_length_of_vocab_word) 255  
入力単語の最大数(MAX_TERMS) 100

* 出力形式  
JSON

_keyに単語、_value(_scoreではない)にコサイン距離が出力されます。
通常のGroongaコマンドと同様に、--output_typeによるjson、xml、tsvの出力指定も可能です。

* 実行例

```
> word2vec_distance "単語" --limit 3 --file_path /var/lib/groonga/db_w2v.bin
[
  [
    0,
    1403607046.4614,
    4.40576887130737
  ],
  [
    [
      3
    ],
    [
      [
        "_key",
        "ShortText"
      ],
      [
        "_value",
        "Float"
      ]
    ],
    [
      "語句",
      0.90013575553894
    ],
    [
      "文中",
      0.886470913887024
    ],
    [
      "辞書",
      0.883183181285858
    ]
  ]
]
```

sentence vectorの例

```
dump_to_train_file Entries title,tag,tags --sentence_vectors 1
word2vec_train --min_count 1 --cbow 1 --sentence_vectors 1
word2vec_distance "doc_id:2" --sentence_vectors 1 --table Entries --column _id,title,tag
[
  [
    0,
    0.0,
    0.0
  ],
  [
    [
      2
    ],
    [
      [
        "_id",
        "UInt32"
      ],
      [
        "title",
        "ShortText"
      ],
      [
        "tag",
        "Tags"
      ]
    ],
    [
      3,
      "Database",
      "Server"
    ],
    [
      1,
      "FulltextSearch",
      "Library"
    ]
  ]
]
```

### ```word2vec_load```

学習済みモデルファイルをロードします。

ファイル名が省略された場合、`{Groongaのデータベースパス}+_w2v.bin`がロードされます。
20個まで同時にロードすることができます。

モデルファイルのサイズにより、ロードは、数秒以上かかることがあります。
Groongaのデータベースを閉じると、自動的にアンロードされます。

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| file_path  | 学習済みモデルファイル | `{Groongaのデータベースパス}+_w2v.bin` |
| binary    | テキスト形式のモデルファイルを使う場合は0 | 1 |

* 出力形式
JSON (true or false)

* 実行例

```
> word2vec_load /var/lib/groonga/db_w2v.bin
[[0,1403598361.75615,4.22779297828674],true]
```

### ```word2vec_unload```

ロードしたモデルファイルをアンロードします。

* 入力形式
なし

* 出力形式
JSON (true)

* 実行例

```
> word2vec_unload
[[0,1403598416.39013,0.00282812118530273],true]
```

## 関数
### ```QueryExpanderWord2vec```
word2vec_distanceを使って動的にクエリ展開をします。

* 実行例

```
>select --table test --match_columns text --query database --query_expander QueryExpanderWord2vec
```

* 設定値  
以下を環境変数で変更可能です。

| env        | description | default      |
|:-----------|:------------|:-------------|
| GRN_WORD2VEC_EXPANDER_LIMIT     | クエリ展開の上限件数 | 3 |
| GRN_WORD2VEC_EXPANDER_THRESHOLD     | クエリ展開用ワードの閾値、1以下の小数を指定 | 0.75 |

* 参考  
[query_expander](https://github.com/groonga/groonga/blob/master/plugins/query_expanders/tsv.c)

## インストール
本プラグインは、Groongaの4.0.3以降のバージョンが必要です。

本プラグインは、RE2ライブラリとMeCabライブラリを利用しています。あらかじめ、これらのライブラリをインストールしてください。CentOSであれば、yumのepelリポジトリでインストールすることができます。

    % yum -y install re2 re2-devel mecab mecab-devel

プラグインをビルドしてインストールしてください。

    % sh autogen.sh
    % ./configure
    % make
    % sudo make install

``./configure --disable-word2vec``の場合、word2vec実行バイナリをインストールしません。

## 使い方

Groongaのデータベースに``word2vec/word2vec``プラグインを登録してください。データベースごとにこの作業が1回だけ必要です。

    % groonga DB
    > register word2vec/word2vec

これにより、Groongaのデータベースに上記のコマンド/関数が登録されて、上記のコマンド/関数が利用できるようになります。

## Author

Naoya Murakami naoya@createfield.com

This program includes the original word2vec code.   

The original word2vec is provided by 2013 Google Inc.  

## License

LGPL 2.1. See COPYING-LGPL-2.1 for details.

The original word2vec code is Apache License 2.0. See COPYING-Apache-License-2.0.  

https://code.google.com/p/word2vec/

Eigen is primarily MPL2 licensed. See vendor/eigen-eigen-b30b87236a1b/COPYING.MPL2 and these links:
