# 这是容器的初始化文件，此脚本会在容器启动后运行，可以在此写上常用包的安装脚本，例如：pip install torch
# 1. 创建conda环境
conda create -n rlver python=3.10 -y
conda activate rlver
# 2. 安装依赖
conda install "mkl<2024.1" "intel-openmp<2024.1" mkl-service mkl_fft mkl_random -c conda-forge -y
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
cd emotion_rlver/code/
pip install -r requirements.txt
cd src/verl
pip install -e .
