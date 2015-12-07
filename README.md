# Groonga word2vec plugin
Groongaにword2vecのコマンド、関数を追加します。

## コマンド
### word2vec_train
Groongaのカラムに格納されたテキストから学習用のファイルを生成し、word2vecで学習させることができます。

カラムからの出力に対してMeCabで分かち書きをし、Groongaのノーマライザ―を使って文字の正規化、およびRE2ライブラリを使って正規表現フィルタをかけることができます。

学習用ファイル名が省略された場合、分かち書きされたテキストファイルが`{Groongaのデータベースパス}+_w2v.txt`に出力されます。現状カラムからの出力は一度かならずテキストファイル化されます。

学習済みモデルファイル名が省略された場合、学習済みモデルファイルが`{Groongaのデータベースパス}+_w2v.bin`に出力されます。

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| table      | Groongaのテーブル | NULL |
| column      | Groongaのカラム名  ``,``区切りで複数指定可  末尾が``_``の場合、スペースを``_``に置換してフレーズ化する  末尾が``/``の場合、形態素解析する  末尾が``*[2-9]``の場合、その回数だけ繰り返し出力する  末尾が``$``の場合、<>のタグと数字と区切り文字``,.;:&^/-#'"()``を削除する | NULL | 
| filter      | Groongaの[スクリプト構文](http://groonga.org/ja/docs/reference/grn_expr/script_syntax.html)で絞り込む | NULL |
| train_file     | 学習用テキストファイル(一時ファイル) |  `{groonga_db}_w2v.txt` |
| output_file   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| normalizer      | Groongaのノーマライザ― NONEの場合なし | NormalizerAuto |
| input_filter   | 入力テキストから除去したい文字列の正規表現(全置換) | NULL |
| input_add_prefix   | 1カラム目の出力の先頭に追加したい文字列 | NULL |
| input_add_prefix_second   | 2カラム目の出力の先頭に追加したい文字列 | NULL |
| mecab_option   | MeCabのオプション Mecab使わない場合NONE | -Owakati |
| save_vovab_file    | save_vocab_file | NULL |
| read_vovab_file    | read_vocab_file | NULL |
| threads    | 学習時のスレッド数 | 12 |
| size     |  次元数 | 100 |
| debug    | debug  学習中の標準出力を止める場合は0 | 2 |
| binary    | モデルファイルをテキスト形式にする場合は0  distanceコマンドはバイナリ形式(1)しか対応していない | 1 |
| cbow    | skip-gramを使う場合は0  cbowを使う場合は1 | 0 |
| alpha    | alpha | 0.025 cbowの場合0.05 |
| window    | 文脈とする前後の幅 | 5 |
| sample    |  高頻度の単語を無視する閾値 | 1e-3 | 
| hs    | 階層的Softmax(高速化手法,ランダム要素有) | 1 |
| negative    | ネガティブサンプリングの単語数(高速化手法,ランダム要素有) | 0 |
| iter    | 学習回数 | 5 |
| min_count    | 単語の最低出現数 | 5 |
| classes    | K-meansクラスタリングする場合は1以上の分類したい数 | 0 |
| is_output_file   | classes>=1でk-meansの出力結果をファイルとする場合1 | 0 |
| sentence_vectors   | sentence vectorを含める場合は1  doc_id:_id(Groongaのtableの_id)の形式で文書ベクトルを追加  (それ以外の単語ベクトルもある) | 0 |

オプションは、通常のGroongaのコマンドと同様に上記の順番で入力する場合は省略することができます。  
上記の順番以外で入力したい場合や省略したい場合は、``--``を先頭につけます。  
たとえば、``word2vec_train --table Logs --column log``

* 出力形式  
JSON

なお、word2vecのデバッグ出力は標準出力されます。 word2vecのエラーは標準エラー出力に出力されます。

* 実行例

```
> word2vec_train Logs log --debug 2
Vocab size: 976190
Words in train file: 219474851
Alpha: 0.000100  Progress: 99.60%  Words/thread/sec: 7.07k
```

k-meansクラスタリングの出力例

```
word2vec_train Entries title,tag,tags --min_count 1 --classes 3
[
  [
    0,
    0.0,
    0.0
  ],
  [
    [
      6
    ],
    [
      [
        "_key",
        "ShortText"
      ],
      [
        "_score",
        "Int32"
      ]
    ],
    [
      "postgresql",
      2
    ],
    [
      "database",
      2
    ],
    [
      "mysql",
      1
    ],
    [
      "mroonga",
      1
    ],
    [
      "fulltextsearch",
      0
    ],
    [
      "groonga",
      0
    ]
  ]
```

### word2vec_distance

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
| limit     | 結果出力の上限件数 | 10 |
| threshold     | 結果出力の閾値、1以下の小数を指定 | -1 |
| normalizer      | Groongaのノーマライザ― | NormalizerAuto |
| term_filter   | 出力をさせない単語にマッチする正規表現(完全一致) | NULL |
| white_term_filter   | 出力をさせる単語にマッチする正規表現(完全一致) | NULL |
| output_filter   | 出力結果から除去したい文字列の正規表現(全置換) | NULL |
| mecab_option   | MeCabのオプション | NULL |
| file_path   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| is_phrase   | スペースを``_``に置換してフレーズ化する場合1 | 0 |
| expander_mode   | 出力形式をクエリ展開用にするかどうかのフラグ<BR>1:クエリ展開 ((query1) OR (query2)) 2:tsv query1\tquery2 | 0 |
| sentence_vectors   | sentence_vectorのみを出力する場合1 | 0 |
| table   | sentence_vectorのdoc_idに対応させるテーブル名 | NULL |
| column   | sentence_vectorのdoc_idに対応して出力するカラム名  ``,``区切りで複数指定可  カラムを出力する場合、最終の_scoreの見かけ上の表示は0になる | _id,_score |
| sortby   | sentence_vectorのdoc_idに対応して出力するカラムのソート  ``,``区切りで複数指定可 | -_score |

* 出力形式  
JSON

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
      "単語",
      2812
    ],
    [
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
]
```

sentence vectorの例

```
word2vec_train Entries title,tag,tags --min_count 1 --cbow 1 --sentence_vectors 1
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

### word2vec_load

学習済みモデルファイルをロードします。

ファイル名が省略された場合、`{Groongaのデータベースパス}+_w2v.bin`がロードされま>す。

モデルファイルのサイズにより、ロードは、数秒以上かかることがあります。
Groongaのデータベースを閉じると、自動的にアンロードされます。

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| file_path  | 学習済みモデルファイル(バイナリ形式) | `{Groongaのデータベースパス}+_w2v.bin` |

* 出力形式
JSON (true or false)

* 実行例

```
> word2vec_load /var/lib/groonga/db_w2v.bin
[[0,1403598361.75615,4.22779297828674],true]
```

### word2vec_unload

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
### QueryExpanderWord2vec
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

