# 【Intel AI × Azure AI】 Azure Custom Visionでモデルを作って、OpenVINOで高速推論する
これはIntelのAI技術とクラウド技術のコラボレーションコンテンツです。
ご存じの通り、Intelのサーバー向けCPUであるXeonはパブリッククラウドのインフラ基盤として多くのクラウドベンダーに利用されております。
そんなXeonですが、近年はディープラーニングワークロードに対して急速にキャッチアップを進めており、最近はXeonのみであっても相当な性能が実現できるようになっています。
さらに、モデルの推論処理がクラウド側からエッジ側へとシフトしているという実情も鑑み、Intelのコンシューマ向けCPUであるCore i3/i5/i7においても性能の向上が可能となっております。
そして、AI開発、特にモデルの開発に欠かせないものがクラウドです。限られた期間の中で膨大なパターンを志向しなければならない学習フェーズにおいて、実質無限のコンピュートリソースをオンデマンドに活用できるクラウドの特性は非常に相性がよく、かつ、近年はモデル開発のための便利なツールも充実してきている点から生産性向上という意味でもその存在感が高まっており、現代のAIエンジニアにとっては必須の学習項目であると考えられます。

そういったわけで、このレッスンでは、クラウドを使ってモデルを作成し、そのモデルをクラウド上（Xeon上）、および、オンプレミス上（Core上）で推論するという、一連の流れをご体験いただきます。
クラウドプラットフォームとして、今回はMicrosoft Azureを使用します。かつ、あらゆる環境（インテルのCPU）においてモデルの推論を高速化するためのツールとしてIntel OpneVINO™　ツールキットを使用します。

- Microsoft Azure：https://azure.microsoft.com/ja-jp/
- Intel OpneVINO™　ツールキット：https://www.intel.co.jp/content/www/jp/ja/internet-of-things/solution-briefs/openvino-toolkit-product-brief.html

## ソフトウェア前提条件
- OS: Windows 10/Ubuntu 18.04にて動作確認
- Python 3.6以上
- pip 19.1.1 以下
- Tensorflow 1.14.0
- Jupyter Notebook
- Intel OpenVINO™ ツールキット 2019R3.1以上
- その他必要なPythonライブラリは手順の中に記載

## 環境構築方法 簡単バージョン(Azure Linux VM編)
Dockerを使用して簡単に環境を構築する方法です。環境構築に興味が無い方はこちらを選択ください。興味がある方は次のセクションの「フルバージョン」を参照ください。
1. [Azure Portal](https://portal.azure.com/)へログインする
1. Azure VMをセットアップする
   
    - [こちら](azurevm_setup_instructions.pdf)の通りに実施ください
    - 以降の操作はAzure Cloud Shell（Bash）上で実施ください
1. Dockerのインストール（参照元は[ここ](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04)）
    - sudo apt update
    - sudo apt install apt-transport-https ca-certificates curl software-properties-common
    - curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    - sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
    - sudo apt update
    - apt-cache policy docker-ce
    - sudo apt install docker-ce
    - sudo usermod -aG docker ${USER}
    - su - ${USER}
    - id -nG
    - sudo usermod -aG docker ai
1. Dockerイメージのダウンロードと実行
    - docker pull hiouchiy/openvino2019r3.1-configured-on-cpu
    - docker run -it -p 8080:8080 hiouchiy/openvino2019r3.1-configured-on-cpu /bin/bash
    - nohup jupyter notebook --ip 0.0.0.0 --allow-root > /dev/null 2>&1 &
1. ローカルPCのWebブラウザを起動し、アドレス欄に「https://AzureVMのパブリックIPアドレス:8080 」と入力
1. Jupyter Notebookのポータル画面にて[Lesson1_AzureCognitiveService_and_OpenVINO_Collaboration.ipynb](Lesson1_AzureCognitiveService_and_OpenVINO_Collaboration.ipynb)をクリックして起動
1. あとはNotebookに従って進める

## 環境構築方法 フルバージョン(Azure Linux VM編)
実習をするための環境をイチから構築する方法です。上記の「簡単バージョン」に比べて手間と時間が掛かりますが、どのようにAI環境を構築するかの詳細が学べます。
1. [Azure Portal](https://portal.azure.com/)へログインする
1. Azure VMをセットアップする
   
    - [こちら](azurevm_setup_instructions.pdf)の通りに実施ください
    - 以降の操作はAzure Cloud Shell（Bash）上で実施ください
1. PIP3のインストール（参照元は[ここ](https://oji-cloud.net/2019/06/16/post-2216/)）
    - mkdir tools
    - cd tools
    - curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
    - sudo apt-get update
    - sudo apt-get install python3-distutils
    - sudo python3 get-pip.py --pip==19.1.1
    - cd ..
1. Jupyter Notebookのインストール（参照元は[ここ](https://qiita.com/JIwatani/items/ae1acb0878610fef3da8)）
    - sudo pip3 install jupyter
    - mkdir ~/.jupyter
    - cd ~/.jupyter/
    - openssl req -x509 -nodes -newkey rsa:2048 -keyout mycert.key -out mycert.pem
        - Country Name (2 letter code) [AU]:**JP**
        - State or Province Name (full name) [Some-State]:**Tokyo**
        - Locality Name (eg, city) []:**Yurakucho**
        - Organization Name (eg, company) [Internet Widgits Pty Ltd]:**intel**
        - Organizational Unit Name (eg, section) []:**ai**
        - Common Name (e.g. server FQDN or YOUR name) []:**student**
        - Email Address []:**皆さんのメールアドレス**
    - ipython3
        - from notebook.auth import passwd
        - passwd()
            - Enter password: **Passw0rd1234**
            - Verify password: **Passw0rd1234**
            - Out[2]: 'sha1:87a95ecd40d0:b00b2037・・・・'　というハッシュが生成されるのメモ帳などにコピーしておく
            - In [3]: exit() 
    - touch ~/.jupyter/jupyter_notebook_config.py
    - vi ~/.jupyter/jupyter_notebook_config.py
        - (※viというのはLinuxにLinuxに搭載されているテキストエディタです。操作方法がやや独特ですが、Linuxを使う上では必須ツールなので、これを機にマスターしましょう。)
		- キーボードの「i」を押すと、入力モードに切り替わるので、以下の内容を記述してください。記述後は、ESCを押して入力モードを完了し、「:wq」の順番で押すことで上書き保存されます。
            ```python
            c = get_config()
            c.NotebookApp.ip = '*'
            c.NotebookApp.open_browser = False
            c.NotebookApp.port = 8080
            c.NotebookApp.password = 'sha1:87a95ecd40d0:b00b2037・・・・'
            #↑こちらには上の操作で生成されたハッシュ値を貼り付ける
            c.NotebookApp.certfile = '/home/ai/.jupyter/mycert.pem'
            c.NotebookApp.keyfile = '/home/ai/.jupyter/mycert.key'
            ```
    - cd ..
1. OpenVINOのインストール（参照元は[ここ](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)）
    - wget https://cs298395642e8d6x4498x8b7.blob.core.windows.net/share/l_openvino_toolkit_p_2019.3.376.tgz
    - tar -xvzf l_openvino_toolkit_p_2019.3.376.tgz
    - cd l_openvino_toolkit_p_2019.3.376/
    - sudo ./install.sh
        - Enter
        - 使用許諾はSpaceを押し続けて最後まで行く
        - accept と入力し、Enter押下
        - 1 と入力しEnterを押下
        - そのままEnter
        - Install Locationが/opt/intelになっていることを確認し、Enter
        - そのままEnter
        - Enter を押下して終了
    - cd /opt/intel/openvino/install_dependencies
    - sudo -E ./install_openvino_dependencies.sh
    - source /opt/intel/openvino/bin/setupvars.sh
    - cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
    - sudo ./install_prerequisites.sh
1. OpenVINOのインストール検証
    - cd /opt/intel/openvino/deployment_tools/demo
    - ./demo_squeezenet_download_convert_run.sh
1. 追加ライブラリをインストール
    - pip3 install pillow --user
    - pip3 install pandas --user
    - pip3 install matplotlib --user
    - pip3 install scipy --user
    - sudo pip3 uninstall tensorflow
    - pip3 install intel-tensorflow==1.14.0 --user
    - sudo apt-get install unzip
1. 本レポジトリをダウンロード
    - mkdir ~/notebook
    - cd ~/notebook/
    - git clone https://github.com/hiouchiy/IntelAI_and_Cloud.git
1. Jupyter Notebookを起動
   
    - nohup jupyter notebook > /dev/null 2>&1 &
1. ローカルPCのWebブラウザを起動し、アドレス欄に「https://AzureVMのパブリックIPアドレス:8080 」と入力
1. Jupyter Notebookのポータル画面にて[Lesson1_AzureCognitiveService_and_OpenVINO_Collaboration.ipynb](Lesson1_AzureCognitiveService_and_OpenVINO_Collaboration.ipynb)をクリックして起動
1. あとはNotebookに従って進める

### おまけ（Tensorflowの置き換え）
1. Intel版のTensorflowをアンインストールし、OSS版のTensorflowをインストールする
    - pip3 uninstall intel-tensorflow
    - pip3 install tensorflow==1.14.0 --user
1. Jupyter Notebookを停止
    - kill 該当するプロセスID
1. Jupyter Notebookを再起動
    - nohup jupyter notebook > /dev/null 2>&1 &

## 環境構築方法(Windows 10編)
1. Python3をインストールする
    - [こちら](install_python3_on_win10.pdf)の通りに実施ください
1. OpenVINO™ ツールキットをインストールする
    - [こちら](install_openvino_on_win10.pdf)の通りに実施ください。
1. Pythonの仮想環境を作成する
    - コマンドプロンプトを起動する
    - python -m venv intelai
    - call intelai\Scripts\activate
1. または、Anacondaの仮想環境を作成する
    - コマンドプロンプトを起動する
    - conda create -n intelai python=3.6 anaconda
    - conda activate intelai
1. 続いて、下記コマンドを順に実行し、ライブラリをインストールする。
    - pip install jupyter
    - pip install requests
    - pip install pillow
    - pip install pandas
    - pip install matplotlib
    - pip install numpy
    - pip install scipy
    - pip install pyyaml
    - pip install tensorflow==1.14.0
    - pip install keras
1. 下記コマンドを実行
    - "C:\Program Files (x86)\IntelSWTools\openvino\bin\setupvars.bat"

### Windows 10上でOpenVINO™ ツールキットのサンプルを動かしてみる
1. 顔検出デモの動かし方
    - mkdir openvino_sample
    - cd openvino_sample
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name face-detection-retail-0004
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name face-detection-adas-0001
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name landmarks-regression-retail-0009
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name face-reidentification-retail-0095
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\demos\python_demos\face_recognition_demo\face_recognition_demo.py" -m_fd intel\face-detection-adas-0001\FP32\face-detection-adas-0001.xml -m_lm intel\landmarks-regression-retail-0009\FP32\landmarks-regression-retail-0009.xml -m_reid intel\face-reidentification-retail-0095\FP32\face-reidentification-retail-0095.xml -fg . -l "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll"
1. 目からビームデモの動かし方
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name head-pose-estimation-adas-0001
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name gaze-estimation-adas-0002
    - python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py" --name facial-landmarks-35-adas-0002
    - [ここ](https://github.com/hiouchiy/OpenVINO_Sample/blob/master/demo1/gaze3.py)からスクリプトをダウンロードし、gaze3.pyというファイル名で保存
    - スクリプトのうち、下記の部分を実際のフォルダーパスで更新
        ```python
        #Absolute Path for CPU Extension lib is needed
        cpu_ext = "C:\\tmp\\cpu_extension_avx2.dll"
        #Absolute Path for downloaded pre-trained model root folder is needed
        model_base_path = 'C:\\tmp\\Transportation' 
        #Paths for each model
        model_det = model_base_path+'\\face-detection-adas-0001\\FP32\\face-detection-adas-0001'
        model_hp = model_base_path+'\\head-pose-estimation-adas-0001\\FP32\\head-pose-estimation-adas-0001'
        model_gaze = model_base_path+'\\gaze-estimation-adas-0002\\FP32\\gaze-estimation-adas-0002'
        model_landmark = model_base_path+'\\facial-landmarks-35-adas-0002\\FP32\\facial-landmarks-35-adas-0002'
        ```
    - python gaze3.py
1. Azure上で作成した犬猫分類モデルをこのWindows10の環境で動かす
   
    - やってみましょう
1. （おまけ）Pythonの仮想環境を削除する
    - call intelaienv\Scripts\deactivate
    - python -m venv --clear intelaienv
