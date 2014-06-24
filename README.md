# Groonga word2vec plugin
Groongaにword2vecのコマンド、関数を追加します。

## コマンド
### word2vec_train
Groongaのカラムに格納されたテキストから学習用のファイルを生成し、word2vecで学習させることができます。

カラムからの出力に対してMeCabで分かち書きをし、Groongaのノーマライザ―を使って文字の正規化、およびRE2ライブラリを使って正規表現フィルタをかけることができます。  

MeCabはオプションを指定することが可能です。たとえば、`-F%m:%f[0]\\0`として出力フォーマットを調整すれば、品詞付の文字列を学習させることができます。単語を基本形に戻して学習させることもできます。  
（なお、その場合は学習後の距離演算のクエリも同様のフォーマットで文字列を入力させる必要があります。）  

Groongaのカラム指定があって学習用ファイル名が省略された場合、分かち書きされたテキストファイルが`{Groongaのデータベースパス}+_w2v.txt`に出力されます。  
学習済みモデルファイル名が省略された場合、分かち書きされたテキストファイルが`{Groongaのデータベースパス}+_w2v.bin`に出力されます。

Groongaのカラムからではないテキストファイルからの学習には対応していません。  
Groongaのカラム指定がなく、且つ、学習用ファイル名が省略された場合、`{Groongaのデータベースパス}+_w2v.txt`が利用されます。  

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| table      | Groongaのテーブル | NULL |
| column      | Groongaのカラム | NULL | 
| train_file     | 学習用テキストファイル |  `{groonga_db}_w2v.txt` |
| output_file   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| normalizer      | Groongaのノーマライザ― | NormalizerAuto |
| input_filter   | 入力テキストから除去したい文字列の正規表現(全置換) | NULL |
| mecab_option   | MeCabのオプション | -Owakati |
| save_vovab_file    | save_vocab_file | NULL |
| read_vovab_file    | read_vocab_file | NULL |
| threads    | threads | 4 |
| size     |  size | 200 |
| debug    | debug | 2 |
| binary    | binary | 1 |
| cbow    | cbow | 0 |
| alpha    | alpha | 0.025 |
| window    | window | 5 |
| sample    | sample | 1e-3 | 
| hs    | hs | 1 |
| negative    | negative | 0 |
| min-count    | min-count | 5 |
| classes    | classes | 0 |

オプションは、通常のGroongaのコマンドと同様に、上記の順番で入力する場合は省略することができます。  
上記の順番以外で入力したい場合や省略したい場合は、``--``を先頭につけます。  
たとえば、``word2vec_train --table Logs --column log``

* 出力形式  
JSON (true or false) 標準出力

なお、現状、word2vecのデバッグ出力はそのまま標準出力されています。

* 実行例

```
> word2vec_train Logs log --input_filter "[0-9]"
Dump column to train file /var/lib/groonga/db_w2v.txt
Starting training using file /var/lib/groonga/db_w2v.txt
Vocab size: 1
Words in train file: 0
```


### word2vec_load

学習済みモデルファイルをロードします。  

ファイル名が省略された場合、`{Groongaのデータベースパス}+_w2v.bin`がロードされます。  

モデルファイルのサイズにより、ロードは、数秒以上かかることがあります。  
Groongaのデータベースを閉じると、自動的にアンロードされます。  
Mroongaの``mroonga_command``は、コマンドごとにデータベースを開いて閉じるため、ロード状態を保つことができません。  

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

### word2vec_distance

入力した単語とベクトル距離が近い単語が出力されます。類似語らしきものを取得することができます。  
出力結果の件数制限、オフセット、閾値、正規表現フィルタをサポートしています。 
たとえば、品詞付で学習しているのであれば、正規表現でフィルタさせることができます。(助詞と助動詞を除く等。)  
学習済みモデルファイルがロードされていない場合、自動的にロードされます。 
ロードされていない場合にファイル名が省略されると、`{Groongaのデータベースパス}+_w2v.bin`がロードされます。  
入力した単語を基本形に戻したり品詞を付けるために、MeCabのオプションを指定することも可能です。

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| term      | 入力単語 | NULL |
| offset      | 結果出力のオフセット | 0 | 
| limit     | 結果出力の上限件数 | 10 |
| threshold     | 結果出力の閾値、1以下の小数を指定 | -1 |
| normalizer      | Groongaのノーマライザ― | NormalizerAuto |
| term_filter   | 出力をさせない単語にマッチする正規表現(完全一致) | NULL |
| output_filter   | 出力結果から除去したい文字列の正規表現(全置換) | NULL |
| mecab_option   | MeCabのオプション | NULL |
| file_path   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| expander_mode   | 出力形式をクエリ展開用にするかどうかのフラグ<BR>1:クエリ展開 ((query1) OR (query2)) 2:tsv query1\tquery2 | 0 |

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

### word2vec_analogy

入力した単語1 - 単語2 + 単語3 = ? の関係に当てはまる単語が出力されます。 類推語らしきものを取得することができます。  
出力結果の件数制限、オフセット、閾値、正規表現フィルタをサポートしています。
たとえば、品詞付で学習しているのであれば、正規表現でフィルタさせることができます。(助詞と助動詞を除く等。)  
学習済みモデルファイルがロードされていない場合、自動的にロードされます。 
ロードされていない場合にファイル名が省略されると、`{Groongaのデータベースパス}+_w2v.bin`がロードされます。  
入力した単語を基本形に戻したり品詞を付けるために、MeCabのオプションを指定することも可能です。

* 入力形式

| arg        | description | default      |
|:-----------|:------------|:-------------|
| positive_term1      | 足し算する単語1 | NULL |
| negative_term      | 引き算する単語 | NULL |
| positive_term2      | 足し算する単語2 | NULL |
| offset      | 結果出力のオフセット | 0 | 
| limit     | 結果出力の上限件数 | 10 |
| threshold     | 結果出力の閾値、1以下の小数を指定 | -1 |
| normalizer      | Groongaのノーマライザ― | NormalizerAuto |
| term_filter   | 出力をさせない単語にマッチする正規表現(完全一致) | NULL |
| output_filter   | 出力結果から除去したい文字列の正規表現(全置換) | NULL |
| mecab_option   | MeCabのオプション | NULL |
| file_path   | 学習済みモデルファイル | `{groonga_db}_w2v.bin` |
| expander_mode   | 出力形式をクエリ展開用にするかどうかのフラグ<BR>1:クエリ展開 ((query1) OR (query2)) 2:tsv query1\tquery2 | 0 |

* 出力形式  
JSON

* 実行例

```
> word2vec_analogy "検索 高速 更新" --limit 3 --file_path /var/lib/groonga/db_w2v.bin

[
  [
    0,1403607202.32257,
    4.45699691772461
  ],
  [
    [
      "検索 高速 更新",
      978.0
    ],
    [
      [
        "低速",
        0.523510456085205
      ],
      [
        "処理速度",
        0.444628149271011
      ],
      [
        "最速",
        0.444386512041092
      ]
    ]
  ]
]
```

## 関数
### QueryExpanderWord2vec
word2vec_distanceを使って動的にクエリ展開をします。

* 実行例

```
>select --table test --match_columns text --query database --query_expander QueryExpanderWord2vec
```

* 設定値  
現状、以下の値は、ハードコーディングされており動的に変更することができません。  
今後、設定ファイル読み込みや環境変数などで変更できるようになるかもしれません。ならないかもしれません。

| arg        | description | default      |
|:-----------|:------------|:-------------|
| limit     | クエリ展開の上限件数 | 3 |
| threshold     | クエリ展開用ワードの閾値、1以下の小数を指定 | 0.75 |
| expander_mode   | 出力形式をクエリ展開用にするかどうかのフラグ | 1 |

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

Groongaのプラグインは共有ライブラリのため、ライブラリパスに注意してください。

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

