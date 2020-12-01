# OpenNMT構築手順

## 1. 変数の設定

以下のファイルの環境変数を設定します。

```sh
opennmt_build_pack/env.source
opennmt_build_pack/opennmt-py/fj-misc/aarch64.multi.conf
```

修正が必須の環境変数は、以下の３つです。
詳細は各ファイルを参照願います。

```sh
PREFIX       # venvを設置したパス(PyTorch環境構築時に設定した値と同じもの).
DATA_DIR     # データをダウンロードし配置するディレクトリ。計算ノードからアクセス可能なディレクトリ.
OPENNMT_PATH # 本ビルドパックのパス.
```

また、環境に合わせて以下のファイルのリソースユニット・リソースグループを設定します。

```sh
opennmt_build_pack/submit_build.sh
opennmt_build_pack/opennmt-py/submit_prepare.sh
opennmt_build_pack/opennmt-py/submit_opennmt.sh
```

## 2. ソースコード/データセットのダウンロード(約20分)

以下を実行してください。


```sh
./dataset.sh  # データセット
./checkout.sh # ソースコード
```

## 3. ビルド & インストール (約1.5時間)

以下を実行してください。

```sh
pjsub submit_build.sh
```

## 4. データ前処理 (約40分)

以下を実行してください。

```sh
cd opennmt-py/
pjsub submit_prepare.sh
```

## 5. 実行

以下を実行してください。スクリプトでは、2プロセス, 4プロセス, 8プロセス, 16プロセスの学習が実行されます。

```sh
pjsub submit_opennmt.sh
```

以上。
