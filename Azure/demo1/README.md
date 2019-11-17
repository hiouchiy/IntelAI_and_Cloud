# 【Azure AIとIntel AIコラボ企画】
# Azure Custom Visionでモデルを作って、OpenVINOで高速推論するサンプル ~サーバーサイド編~

## 前提条件
- Python 3.6以上
- Tensorflow 1.13.1以上
- Jupyter Notebook
- Intel OpenVINO™ Toolkit 2019R3.1以上
- OS: Windows 10/Ubuntu 16.04にて動作確認

## 環境構築方法(Azure Linux VM編)
1. [Azure Portal](https://portal.azure.com/)へログインする
1. [こちら](azurevm_setup_instructions.pdf)の通りにAzure VMをセットアップする
1. PIP3のインストール(参照元：https://oji-cloud.net/2019/06/16/post-2216/)
    - mkdir tools
    - cd tools
    - curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
    - sudo apt-get update
    - sudo apt-get install python3-distutils
    - sudo python3 get-pip.py
    - cd ..
1. Jupyter Notebookのインストール（参照元：https://qiita.com/JIwatani/items/ae1acb0878610fef3da8 ）
    - sudo pip3 install jupyter
    - mkdir ~/.jupyter
    - cd ~/.jupyter/
    - openssl req -x509 -nodes -newkey rsa:2048 -keyout mycert.key -out mycert.pem
        - Country Name (2 letter code) [AU]:**JP**
        - State or Province Name (full name) [Some-State]:**Tokyo**
        - Locality Name (eg, city) []:**Shinjuku**
        - Organization Name (eg, company) [Internet Widgits Pty Ltd]:**tech.c**
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
        - 以下の内容を記述(iで入力モード)し、保存(Esc押下後、:wqで上書き保存)
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
1. OpenVINOのインストール（参照元：https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html）
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
    - pip3 uninstall tensorflow
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
1. コマンドプロンプトを開き、下記コマンドにてライブラリを追加インストールする。
    - pip install pillow --user
    - pip install pandas --user
    - pip install matplotlib --user
    - pip install numpy --user
    - pip install scipy --user
    - pip install opencv-python --user
    - pip install tensorflow==1.14.0 --user
1. コマンドプロンプトを一度閉じ、再度コマンドプロンプトを開き、下記コマンドを実行
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