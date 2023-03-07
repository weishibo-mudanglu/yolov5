import random
from re import T
import shutil
import json
import cv2
import os
import time
# from turtle import pos
import imagesize
import sys
from tqdm import tqdm
from pathlib import Path
from path_collect import *
# label_dict = {'class_count': 0}
# label_dict = {'轿车':'0', '双轮非机动车':'1', '聚集车辆或模糊类别':'2', '面包车':'3', '卡车':'4',
#                   '客车':'5', '货车':'6', '工程车':'7', '公交车':'8', '三轮车':'9', '渣土车':'10', '罐车':'11', '叉车':'12',
#                   '自行车':'13', '搅拌车':'14', '清洁车':'15', '垃圾车':'16',  '挖掘机':'17', '重型拖车':'18', '消防车':'19', '吊车':'20'}
# label_dict = {'class_count': 11, '轿车': '0', '双轮非机动车': '1', '三轮车': '2','渣土车': '3', 
#                             '货车': '4', '客车': '5', '卡车': '6', '罐车': '7', '工程车': '8', '清洁车': '9', '聚集车辆或模糊类别': '10'}
# label_dict = {'class_count': 4, '灭火器': '0', '推车式灭火器': '1', '手推式灭火器': '1', '路锥': '2', '作业指示牌-A字形': '3'}
# label_dict = {'class_count': 2, '\u706b': '0', '\u70df\u96fe': '1'}#\u706b 火 \u70df\u96fe烟雾
label_dict = {'class_count': 2, 'helmet': '0','none':'0'}

def make_from_json(json_dir, txt_dir, label_dict):
    json_path_list = list(json_dir.iterdir())
    # json_path_list = list(json_dir.glob("187358931172460827*"))

    for json_path in tqdm(json_path_list):
        label_name = json_path.name.split('.')[0] + '.txt'
        img_shape = json_path.name.split('.')[0] + '.json'
        img_shape_path = json_dir.parent/'shape'/img_shape

        if not img_shape_path.is_file():
            print(img_shape_path.name)
            continue

        with open(str(img_shape_path)) as f:
            label_str = f.read()
            shape_dict = json.loads(label_str)
            x = shape_dict['x']
            y = shape_dict['y']
        # try:
        with open(str(json_path)) as f:
            label_str = f.read()
            # label_json = json.loads(label_str)['outputs']['object']
            label_json = json.loads(label_str)['entityAnn']

            xmin_old = 0
            xmax_old = 0
            ymin_old = 0
            ymax_old = 0 
            if not os.path.exists(str(txt_dir / label_name)):
                with open(str(txt_dir / label_name), 'a') as f:
                # if True:
                    for label_tagAnn in label_json:
                        #部分json文件在这里没有meaningful标签，如果没有会返回一个包含坐标的列表，如果有会返回字典
                        posinfo = json.loads(label_tagAnn['posInfo'][0][6])
                        if(isinstance(posinfo,dict)):
                            posinfo = posinfo['meaningful']
                        xmin = min(min(posinfo[0]['x'], posinfo[1]['x']),min(posinfo[2]['x'], posinfo[3]['x']))
                        xmax = max(max(posinfo[0]['x'], posinfo[1]['x']),max(posinfo[2]['x'], posinfo[3]['x']))
                        ymin = min(min(posinfo[0]['y'], posinfo[1]['y']),min(posinfo[2]['y'], posinfo[3]['y']))
                        ymax = max(max(posinfo[0]['y'], posinfo[1]['y']),max(posinfo[2]['y'], posinfo[3]['y']))
                        if xmax == xmin or ymax == ymin:
                            # print('点', label_name)
                            continue

                        xmin = xmin if xmin > 0 else 0
                        xmax = xmax if xmax < x else x - 1
                        ymin = ymin if ymin > 0 else 0
                        ymax = ymax if ymax < y else y - 1

                        if xmax == xmax_old and xmin == xmin_old and ymax == ymax_old and ymin == ymin_old:
                            # print('重复', label_name)
                            continue

                        xmin_old = xmin
                        xmax_old = xmax
                        ymin_old = ymin
                        ymax_old = ymax
                        # assert xmax > xmin, f"xmax:{xmax} < xmin:{xmin}, label_name:{label_name}"
                        # assert ymax > ymin, f"ymax:{ymax} < ymin:{ymin}, label_name:{label_name}"
                        xc = (xmin + xmax) / 2 / x
                        yc = (ymin + ymax) / 2 / y
                        width = (xmax - xmin) / x
                        height = (ymax - ymin) / y

                        if label_tagAnn['name'] in label_dict:
                            class_idx = label_dict[label_tagAnn['name']]
                            f.write(str(class_idx) + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(width) + ' ' + str(height) + '\n')
                        # else:
                        #     label_dict[label_tagAnn['name']] = str(label_dict['class_count'])
                        #     class_idx = label_dict['class_count']
                        #     label_dict['class_count'] += 1
                        # f.write(str(class_idx) + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(width) + ' ' + str(height) + '\n')
    # print(label_dict)
                
#读取json之前先读取图片生成尺寸的json文件
def make_image_shape_json(image_dir):
    shape_json_dir = image_dir.parent/'shape'
    shape_json_dir.mkdir(exist_ok=True)

    image_path_list = list(image_dir.glob('*.jpeg'))
    for image_path in tqdm(image_path_list):
        json_path = shape_json_dir/ (image_path.name.split('.')[0] + '.json')
        if json_path.is_file():
            continue
        try:
            #imagesize库有些图片会直接返回[-1，-1]未排查出具体原因，这里做一个判断，如果出现则加载图片到内存获得shape
            x,y = imagesize.get(str(image_path))
            if (x==-1 or y==-1):
                image = cv2.imread(str(image_path))
                x = image.shape[0]
                y = image.shape[1]
        except:
            print(image_path)
        json_dict = {'x':x,'y':y}
        shape_json = json.dumps(json_dict)
        with open(str(json_path), 'a') as f:
            f.write(shape_json)

def count_img_shape(shape_path):
    shapejsonlist = list(shape_path.glob('*.json'))
    count = {}
    for shape_json in tqdm(shapejsonlist):
        with open(str(shape_json)) as f:
            label_str = f.read()
            shape_dict = json.loads(label_str)
            x = shape_dict['x']
            y = shape_dict['y']
            if str(x)+'x' in count:
                count[str(x)+'x'] += 1
            else:
                count[str(x)+'x'] = 1
            if str(y)+'y' in count:
                count[str(y)+'y'] += 1
            else:
                count[str(y)+'y'] = 1
    print(count)

def valid_txt(ann_file):
    ann_files = list(ann_file.glob('*.txt'))
    for file_path in tqdm(ann_files):
        img_path = file_path.parent.parent / 'images' / (file_path.name.split('.txt')[0] + '.jpg')
        img = cv2.imread(str(img_path))
        x = img.shape[1]
        y = img.shape[0]
        with open(file_path) as f:
            for filestr in f.readlines():
                number = filestr.strip().split(' ')
                samples = list()
                for i in number:
                    samples.append(float(i))
                label = str(samples[0])
                x1 = int(samples[1] * x - samples[3] * x / 2)
                y1 = int(samples[2] * y - samples[4] * y / 2)
                x2 = x1 + int(samples[3] * x)
                y2 = y1 + int(samples[4] * y)
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255), 2)
                cv2.putText(img, label, (int(x1), int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        img = cv2.resize(img, (1080, 720))
        cv2.imshow('img', img)
        cv2.waitKey(0)

def valid_txt_save_img(ann_file):
    if os.path.exists('/data/liutianchi/code/yolov5-3.0/inference/show/'):
        shutil.rmtree('/data/liutianchi/code/yolov5-3.0/inference/show/')
    Path('/data/liutianchi/code/yolov5-3.0/inference/show/').mkdir(exist_ok=True)
    
    ann_files_all = list(ann_file.glob('*.txt'))
    ann_files = random.sample(ann_files_all, 100)
    for file_path in tqdm(ann_files):
        img_path = file_path.parent.parent / 'images' / (file_path.name.split('.txt')[0] + '.jpg')
        img = cv2.imread(str(img_path))
        x = img.shape[1]
        y = img.shape[0]
        with open(file_path) as f:
            for filestr in f.readlines():
                number = filestr.strip().split(' ')
                samples = list()
                for i in number:
                    samples.append(float(i))
                label = str(samples[0])
                x1 = int(samples[1] * x - samples[3] * x / 2)
                y1 = int(samples[2] * y - samples[4] * y / 2)
                x2 = x1 + int(samples[3] * x)
                y2 = y1 + int(samples[4] * y)
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255), 2)
                cv2.putText(img, label, (int(x1), int(y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        img = cv2.resize(img, (1080, 720))
        cv2.imwrite('/data/liutianchi/code/yolov5-3.0/inference/show/'+img_path.name,img)

#×××××××××××××××××××××××××××××××
#找出含有错误标签的文件并放入error文件夹
#×××××××××××××××××××××××××××××××
def find_error_label(txt_dir):
    ann_files=os.listdir(txt_dir)
    txt_dir_ =  Path(txt_dir)
    for file_path in tqdm(ann_files):
        txtpath =  txt_dir_.parent / 'txts' / Path(os.path.splitext(file_path)[0]+'.txt')
        imgpath =  txt_dir_.parent / 'images' / Path(os.path.splitext(file_path)[0]+'.jpg')
        shapepath = txt_dir_.parent / 'shape' / Path(os.path.splitext(file_path)[0]+'.json')
        labelpath = txt_dir_.parent / 'labels' / Path(os.path.splitext(file_path)[0]+'.json')
        totxtpath = txt_dir_.parent.parent / 'error_image' / txt_dir_.parts[-2] / 'txts' 
        toimgpath = txt_dir_.parent.parent / 'error_image' / txt_dir_.parts[-2] / 'images' 
        toshapepath = txt_dir_.parent.parent / 'error_image' / txt_dir_.parts[-2] / 'shape'  
        tolabelpath = txt_dir_.parent.parent / 'error_image' / txt_dir_.parts[-2] / 'labels' 
        totxtpath_ = Path(totxtpath)
        totxtpath_.mkdir(exist_ok=True, parents=True)
        toimgpath_ = Path(toimgpath)
        toimgpath_.mkdir(exist_ok=True, parents=True)
        toshapepath_ = Path(toshapepath)
        toshapepath_.mkdir(exist_ok=True, parents=True)
        tolabelpath_ = Path(tolabelpath)
        tolabelpath_.mkdir(exist_ok=True, parents=True)
        with open(txtpath,'r') as txt:
            lines = txt.readlines()
        moveFlage = False
        for line in lines:
            nums = [float(numStr) for numStr in line.replace("\n", '').split(' ')[1:]]
            if '-' in line:
                moveFlage = True
            for num in nums:
                if num>=1.0:
                    moveFlage=True

        if moveFlage:
            shutil.move(str(txtpath),str(totxtpath))
            shutil.move(str(imgpath),str(toimgpath))
            shutil.move(str(shapepath),str(toshapepath))
            shutil.move(str(labelpath),str(tolabelpath))

#×××××××××××××××××××××××××××××××××××××
#按2/10的比例取所有数据集的部分数据为测试集
#×××××××××××××××××××××××××××××××××××××
def move_trian2test(txt_dir):
    ann_files=os.listdir(txt_dir)
    txt_dir_ =  Path(txt_dir)
    for file_path in tqdm(ann_files):
        txtpath =  txt_dir_.parent / 'txts' / Path(os.path.splitext(file_path)[0]+'.txt')
        imgpath =  txt_dir_.parent / 'images' / Path(os.path.splitext(file_path)[0]+'.jpg')
        temp_file = Path(imgpath)
        if not temp_file.is_file():
            imgpath = txt_dir_.parent / 'images' / Path(os.path.splitext(file_path)[0]+'.jpeg')
        shapepath = txt_dir_.parent / 'shape' / Path(os.path.splitext(file_path)[0]+'.json')
        labelpath = txt_dir_.parent / 'labels' / Path(os.path.splitext(file_path)[0]+'.json')
        temp_file = Path(labelpath)
        if not temp_file.is_file():
            labelpath = txt_dir_.parent / 'labels' / Path(os.path.splitext(file_path)[0]+'.xml')
        totxtpath = txt_dir_.parent.parent / 'test_image' / txt_dir_.parts[-2] / 'txts' 
        toimgpath = txt_dir_.parent.parent / 'test_image' / txt_dir_.parts[-2] / 'images' 
        toshapepath = txt_dir_.parent.parent / 'test_image' / txt_dir_.parts[-2] / 'shape'  
        tolabelpath = txt_dir_.parent.parent / 'test_image' / txt_dir_.parts[-2] / 'labels'
        totxtpath_ = Path(totxtpath)
        totxtpath_.mkdir(exist_ok=True, parents=True)
        toimgpath_ = Path(toimgpath)
        toimgpath_.mkdir(exist_ok=True, parents=True)
        toshapepath_ = Path(toshapepath)
        toshapepath_.mkdir(exist_ok=True, parents=True)
        tolabelpath_ = Path(tolabelpath)
        tolabelpath_.mkdir(exist_ok=True, parents=True)
        flag = random.randint(0,9)
        if (flag < 2):
            shutil.move(str(txtpath),str(totxtpath))
            shutil.move(str(imgpath),str(toimgpath))
            shutil.move(str(shapepath),str(toshapepath))
            shutil.move(str(labelpath),str(tolabelpath))
def create_false_txt_label(img_dir,txt_dirs):
    img_list = img_dir.glob('*.jpeg')
    for img_path in tqdm(img_list):
        name = img_path.name.split('.jpeg')[0] + '.txt'
        txt_name = img_path.parent.parent / 'txts' / name
        with open(str(txt_name), 'a') as f:
            pass
if __name__ == '__main__':
    data_dir_list = temp_img_dirs
    for data_dir in data_dir_list:
        print("正在处理文件夹："+data_dir)
        json_dir = Path(data_dir) / 'labels'
        images_dir = Path(data_dir) / 'images'

        # shape_dir = Path('/data/liutianchi/datasets/vehicle/generalVehicleDetectionDataset_train_850k/shape')
        # ann_file = Path('/data/liutianchi/datasets/vehicle/generalVehicleDetectionDataset_test_60k/txts')
        txt_dir = json_dir.parent / 'txts'
        # print(txt_dir)
        # txt_dir.mkdir(exist_ok=True)
        # json_dir.mkdir(exist_ok=True)

        # valid_txt_save_img(Path('/data/liutianchi/datasets/vehicle/通用车辆检测数据集_自动训练平台_10w_平视/txts'))
        # valid_txt(ann_file)
        # make_image_shape_json(images_dir)
        # count_img_shape(shape_dir)
        # make_from_json(json_dir, txt_dir, label_dict)
        # create_false_txt_label(images_dir,txt_dir)
        # find_error_label(txt_dir)
        move_trian2test(txt_dir)


