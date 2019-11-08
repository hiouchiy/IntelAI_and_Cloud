# 【Azure AIとIntel AIコラボ企画】
# Azure Custom Visionでモデルを作って、OpenVINOで高速推論するサンプル

## 前提条件
- Python 3.6以上
- Tensorflow 1.13.1以上
- Jupyter Notebook
- Intel OpenVINO™ Toolkit 2019R3.1以上
- OS: Windows 10/Ubuntu 16.04にて動作確認

## 環境構築方法
1. [Azure Portal](https://portal.azure.com/)へログインする
1. ホーム画面左上あたりの「リソースの作成」をクリック
1. 

## 動かし方
- git clone https://github.com/hiouchiy/OpenVINO_Sample.git
- cd OpenVINO_Sample
- OpenVINOの環境変数周りのセットアップ
    - Windows: "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"
    - Ubuntu: source /opt/intel/openvino/bin/setupvars.sh
- jupyter notebook
- Jupyter Notebookのポータル画面にて「AzureCognitiveService_and_OpenVINO_Collaboration.ipynb」をクリックして起動
- あとはNotebookに従って進める
