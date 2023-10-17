# Det-data-datageneration-base-SAM
基于ppseg-sam的半自动数据生成工具，由sam分割出目标mask，自动对图片集进行贴mask和直方图匹配，数量，缩放尺寸，位置，翻转可进行参数控制

## 1.创建ppseg环境
### 1.1安装paddlepaddle-gpu（>2.4）
url：https://www.paddlepaddle.org.cn/
# 在Python解释器中顺利执行如下命令
>>> import paddle
>>> paddle.utils.run_check()
# 如果命令行出现以下提示，说明PaddlePaddle安装成功
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

### 1.2下载安装paddleseg
git-url:  
Gitee-url：https://gitee.com/paddlepaddle/PaddleSeg.git
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
### 1.3检验安装成功
sh tests/install/check_predict.sh
