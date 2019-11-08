# MobileNetの学習と推論
これはMobileNetのモデルを構築し、それを推論実行するサンプルです。
モデルの構築には、Keras（Tensorflowバックエンド）を使用します。
モデルの推論には、OpenVINOを使用します。

## 推奨環境
- Windows 10
or
- Ubuntu 16.04

## 前提ソフトウェア
- environment.yml を参照してください。
- OpenVINO 2019R2.0.1以上

## 学習編
- コマンドプロンプトを起動
- このリポジトリをクローンしたフォルダへ移動
- 「jupyter notebook」とタイプ
- WebブラウザでJupyter Notebookが起動したら、Demo_Training_MobileNet.ipynb をクリック
- Notebookの内容（英語）を上から順番に実行

## 推論編
- コマンドプロンプトを開き、下記スクリプトを実行
    - Windowsの場合：OpenVINOインストールフォルダ/bin/setupvars.bat
    - Linuxの場合：source OpenVINOインストールフォルダ/bin/setupvars.sh
- このリポジトリをクローンしたフォルダへ移動
- 「jupyter notebook」とタイプ
- WebブラウザでJupyter Notebookが起動したら、Demo_Training_MobileNet.ipynb をクリック
- Notebookの内容（英語）を上から順番に実行

