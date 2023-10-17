# Det-data-datageneration-base-SAM
基于ppseg-sam的半自动数据生成工具，由sam分割出目标mask，自动对图片集进行贴mask和直方图匹配，数量，缩放尺寸，位置，翻转可进行参数控制

## 1.创建ppseg环境
### 1.1安装paddlepaddle-gpu（>2.4）
> url：https://www.paddlepaddle.org.cn/
### 在Python解释器中顺利执行如下命令
```
import paddle 
paddle.utils.run_check()
```
*如果命令行出现以下提示，说明PaddlePaddle安装成功*
*PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.*

### 1.2下载安装paddleseg
> git-url:  
> Gitee-url：https://gitee.com/paddlepaddle/PaddleSeg.git
```
cd PaddleSeg
pip install -r requirements.txt
pip install -v -e .
```
### 1.3检验安装成功
```
sh tests/install/check_predict.sh
```
### 1.4 替换脚本
下载本项目 *scripts* 文件夹对 *PaddleSeg\contrib\SegmentAnything\scripts* 文件夹进行替换
### 1.5 文件结构
在*PaddleSeg\contrib\SegmentAnything*下新建sample文件夹
> sample
> > Masks (sam 抠图存放地址)

> > Data （原图片地址）

> > Images （进行抠图的目标文件地址,脚本会读取子文件夹名并在生成写入json时使用该名称作为label）

> > > car

> > > fish

## 2.参数设置
### 2.1 01_promt_predict_mask.py 参数设置
#### 主程序
    *待分割图像集地址*
    inputspath = abs path to sample/Images  
    *写入json文件时的图片ID起始值，用于合并数据集时直接复制，应设置为-1或（其他数据集图像个数-1）*
    image_num = -1
    *写入json文件时的Annotation起始值，用于合并数据集时直接复制，应设置为-1或（其他数据集ann个数-1）*
    object_num = -1
#### part 1
  使用sam抠图，程序启动后鼠标点击并拖动画框，使目标至于其中，松手开始分割抠图，蓝框变红可以进行下一次操作，关闭后载入下一张图，允许空操作
#### part2
  贴图,随机参数设置在函数*random_point_and_scala（）*中。
#### part 3
  写入json文件（coco style），随机贴图的参数存放在labels-info.txt中
### 2.2 _lzy2coco.py
    line 12: labels_list = ['fish','car']
    line 13: label_to_num = {'fish':1, 'car':2}
    对应Images子文件夹，用于coco格式json中categories字段的映射，如果是新类别，应该设置不同的映射值
## 3.运行
run 01_promt_predict_mask.py
    
