# 【Azure AIとIntel AIコラボ企画】
# Azure Custom Visionでモデルを作って、OpenVINOで高速推論するサンプル

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
            - Out[2]: 'sha1:87a95ecd40d0:b00b2037・・・・'　というハッシュが生成される
            - In [3]: exit() 
    - touch ~/.jupyter/jupyter_notebook_config.py
    - vi ~/.jupyter/jupyter_notebook_config.py
        - 以下の内容を記述(iで入力モード)し、保存(Esc押下後、:wqで上書き保存)
            c = get_config()
            c.NotebookApp.ip = '*'
            c.NotebookApp.open_browser = False
            c.NotebookApp.port = 8080
            c.NotebookApp.password = 'sha1:87a95ecd40d0:b00b2037・・・・'
            c.NotebookApp.certfile = '/home/hiouchiy/.jupyter/mycert.pem'
            c.NotebookApp.keyfile = '/home/hiouchiy/.jupyter/mycert.key'
1. OpenVINOのインストール（参照元：https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html）
    - wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16057/l_openvino_toolkit_p_2019.3.376.tgz
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
    - pip3 install keras --user
1. 本レポジトリをダウンロード
    - mkdir ~/notebook
    - cd ~/notebook/
    - git clone https://github.com/hiouchiy/IntelAI_and_Cloud.git
1. Jupyter Notebookを起動
    - jupyter notebook & 
    - または 
    - nohup jupyter notebook > /dev/null 2>&1 &
1. ローカルPCのWebブラウザを起動し、アドレス欄に「https://AzureVMのパブリックIPアドレス:8080 」と入力
1. Jupyter Notebookのポータル画面にて「IntelAI_and_Cloud/Azure/demo1/AzureCognitiveService_and_OpenVINO_Collaboration.ipynb」をクリックして起動
1. あとはNotebookに従って進める
