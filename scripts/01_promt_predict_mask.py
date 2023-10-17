import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import paddle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry
import time
from pathlib import Path
from skimage import io,exposure
import glob
import os
import random



model_link = {
    'vit_h':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'vit_l':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'vit_b':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams"
}


'''SAM part'''
'''SAM part'''
'''SAM part'''
global img
global point1,point2
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image1 = (mask.reshape(h, w, 1) * 255)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return np.array(mask_image1,dtype=np.uint8)

def sam_part(model_type='vit_l',input_path='',box=''):
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    input_path = input_path
    box = box
    if box is not None:
        box = box.split(' ')
        box = np.array([[box[0], box[1]], [box[2], box[3]]])

    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = sam_model_registry[model_type](
        checkpoint=model_link[model_type])
    predictor = SamPredictor(model)
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        mask_input=None,
        multimask_output=True, )

    mask = show_mask(masks[0], plt.gca())
    return mask

'''Mouse Envent'''
'''Mouse Envent'''
'''Mouse Envent'''
def on_mouse(event,x,y,flags,path):
    global img,point1,point2
    img2=img.copy()
    if event==cv2.EVENT_LBUTTONDOWN:#左键点击
        point1=(x,y)
        cv2.circle(img2,point1,10,(0,255,0),5)
        cv2.imshow('image',img2)

    elif event==cv2.EVENT_MOUSEMOVE and (flags&cv2.EVENT_FLAG_LBUTTON):#移动鼠标，左键拖拽
        cv2.rectangle(img2,point1,(x,y),(255,0,0),15)#需要确定的就是矩形的两个点（左上角与右下角），颜色红色，线的类型（不设置就默认）。
        cv2.imshow('image',img2)

    elif event==cv2.EVENT_LBUTTONUP:#左键释放
        point2=(x,y)
        cv2.rectangle(img2,point1,point2,(0,0,255),5)#需要确定的就是矩形的两个点（左上角与右下角），颜色蓝色，线的类型（不设置就默认）。
        cv2.imshow('image',img2)
        min_x=min(point1[0],point2[0])
        min_y=min(point1[1],point2[1])
        width=abs(point1[0]-point2[0])
        height=abs(point1[1]-point2[1])
        # cut_img=img[min_y:min_y+height,min_x:min_x+width]
        points = str(min_x) + ' ' + str(min_y) + ' ' + str(min_x+width) + ' ' + str(min_y+height)
        print('-------------------------------像素分割中---------------------------------------')
        mask = sam_part(input_path=input, box=points)
        clip_img = img[min_y:min_y+height,min_x:min_x+width]
        clip_mask = mask[min_y:min_y+height,min_x:min_x+width]
        log = str(time.time())
        bname = os.path.basename(path)
        path = path.replace(bname,'')
        outpath = path.replace('Images','masks')
        maskname = outpath.split('\\')[-2]
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        print('-------------------------------分割完成，mask保存中---------------------------------------')
        cv2.imwrite(os.path.join(outpath, maskname + "_mask_" + log + '.jpg'), clip_mask)
        cv2.imwrite(os.path.join(outpath, maskname + "_img_" + log + '.jpg'), cv2.bitwise_and(clip_img,np.concatenate([clip_mask, clip_mask, clip_mask], 2)))
        print('-------------------------------mask保存完成---------------------------------------')


def box_index(path):
    global img
    img=cv2.imread(path)
    cv2.namedWindow('image',0)
    cv2.resizeWindow('image', 900, 700)   # 自己设窗口图片的大小
    cv2.setMouseCallback('image',on_mouse,path)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# single img process
class Mask_Process():
    '''
    1.贴图 class：Mask_Process
    2.记录坐标 Mask_Process.random_point_and_scala()
    3.导出标注文件（CoCo） 检查格式
    '''
    def __init__(self,img_path,mask_path):
        # self.root = rootpath
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        self.mask_path = mask_path
        # self.mask_name = self.mask_path.split('\\')[-1]
        self.H,self.W,_ = self.img.shape

    # process params for single img
    def random_point_and_scala(self):
        all_mask = []
        '''debug'''
        '''debug'''
        mask_lib = list(Path(self.mask_path).rglob('*_mask_*'))
        for _ in range(random.randint(4,9)):
            #根据mask和Img的shape设置缩放阈值
            mask_path = os.path.join(self.mask_path, str(random.choice(mask_lib)))
            mask_name = mask_path.split('\\')[-2]
            scala = random.uniform(0.3,3)
            Fxy = [scala*random.uniform(0.6,1.6),scala*random.uniform(0.6,1.6)]
            point = [random.randint(0,self.H),random.randint(0,self.W)]
            single_mask = {"imgpath": self.img_path, "maskpath":mask_path, "Fxy": Fxy,  "point": point, "maskname":mask_name, "affine":True}
            all_mask.append(single_mask)

        return  all_mask
    # single mask process
    def paste(self,single_mask:dict):
        # mask颜色反转，贴图区域为0
        clip_img = cv2.imread(single_mask["maskpath"].replace('mask','img'))
        re_mask = cv2.bitwise_not(cv2.imread(single_mask["maskpath"]))
        # 放缩变换
        re_mask = cv2.resize(re_mask,None,fx=single_mask["Fxy"][0],fy=single_mask["Fxy"][1],interpolation=cv2.INTER_LINEAR)
        clip_img = cv2.resize(clip_img,None,fx=single_mask["Fxy"][0],fy=single_mask["Fxy"][1],interpolation=cv2.INTER_LINEAR)
        if single_mask["affine"]:
            re_mask,clip_img = self.transform(re_mask,clip_img)
        y,x = single_mask["point"]
        h,w,_ = re_mask.shape
        if y+h < self.H and x+w < self.W:
            # mask与对应位置做与运算
            self.img[y:y+h,x:x+w,:] = cv2.bitwise_and(self.img[y:y+h,x:x+w,:],re_mask)

            '''色调融合'''
            clip_img = exposure.match_histograms(clip_img, self.img, multichannel=True)
            clip_img = cv2.bitwise_and(clip_img, cv2.bitwise_not(re_mask))
            # 贴图与对应位置做加法
            self.img[y:y + h, x:x + w, :] = cv2.add(self.img[y:y + h, x:x + w, :],clip_img)
            return 1
        else:
            return 0

    # single img all mask process
    def paste_all(self,all_mask:dict):
        with open('labels-info.txt','a+') as f:
            for single_mask in all_mask:
                flag = self.paste(single_mask)
                if flag:
                    f.write(str(single_mask) + '\n')
                    print(single_mask)
        # save_img
        cv2.imwrite(self.img_path,self.img)

    def transform(self,mask,clip_img):
        # 获取图像宽度和高度
        # height, width = clip_img.shape[:2]
        # # 定义旋转角度和旋转中心
        # angle = random.randint(-100,100)  # 旋转角度
        # center = (width // 2, height // 2)  # 旋转中心
        # # 计算旋转矩阵
        # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # # 进行图像旋转
        # clip_img = cv2.warpAffine(clip_img, rotation_matrix, (width, height))
        revers = random.choice([-1,0,1,None])
        if revers:
            # 进行水平竖直水平翻转
            mask = cv2.flip(mask, revers)
            clip_img = cv2.flip(clip_img, revers)
        return mask,clip_img

def write2coco(txtfile):

    pass


if __name__ == "__main__":



    # per part can run solo, you can #part1 and run part2,3 when you already segment mask
    # per part can run solo, you can #part1 and run part2,3 when you already segment mask

    inputspath = r'F:\server_project\PaddleSeg\contrib\SegmentAnything\sample\Images'
    image_num = -1
    object_num = -1


    '''part 1: segment mask'''
    '''part 1: segment mask'''
    # imgs = list(Path(inputspath).rglob('*.png'))
    # for input in imgs:
    #     input = os.path.join(inputspath,str(input))
    #     box_index(input)

    '''part 2: generation dataset'''
    '''part 2: generation dataset'''
    # imgpath = inputspath.replace('Images','Data')
    # maskpath = inputspath.replace('Images','Masks')
    # imgs = list(Path(imgpath).rglob('*'))
    # for img in imgs:
    #     mask_procss = Mask_Process(str(img),maskpath)
    #     all_mask = mask_procss.random_point_and_scala()
    #     mask_procss.paste_all(all_mask)
    #     print(all_mask)

    '''part 3: write 2 .json with coco style'''
    '''part 3: write 2 .json with coco style'''
    from _lzy2coco import deal_json_custom,MyEncoder
    label_info = r'labels-info.txt'
    train_data_coco = deal_json_custom(label_info,image_num=image_num,object_num=object_num)
    train_json_path = 'instance_train.json'
    json.dump(
        train_data_coco,
        open(train_json_path, 'w'),
        indent=4,
        cls=MyEncoder)