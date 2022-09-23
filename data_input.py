import json
import os
import random
import numpy as np
from tqdm import trange
from PIL import Image
from torchvision import transforms
import torch.utils.data
import torch
import copy
train_valid_rate = 0.8
#from translate import Translator


def get_data_img_and_text(img_path,type_of_size):
    train_path = img_path + '/train'
    test_path = img_path + '/test'
    en_ch_json = {}
    with open('dirty_cn_en.json','r',encoding = 'utf-8') as file:
        en_ch_json = json.load(file)
    
    train_all_json_path = img_path + '/train_all.json'
    test_all_json_path = img_path + '/test_all.json'
    #special_labels_cn  = ["男茶色病号服","男蓝白病号服","女粉病号服","土红","矢车菊蓝","灰色条【送腰带】","红色条【送腰带】","深杏"]
    #special_labels_en  = ["man tea color","man blue white","women pink","earthy red","Cornflower blue","Gray strip belt","Red strip belt","Deep apricot"]
    #special_labels_en = ["男茶色病号服", "男蓝白病号服", "女粉病号服", "土红", "矢车菊蓝", "灰色条【送腰带】", "红色条【送腰带】", "深杏"]
    with open(train_all_json_path, 'r', encoding='utf-8') as file:
        train_all_json = json.load(file)

    with open(test_all_json_path,'r',encoding='utf-8') as file:
        test_all_json = json.load(file)

    train_dirs = os.listdir(train_path)#目录train
    test_dirs = os.listdir(test_path)#目录test
    train_data = []#三元组形式，[optinal_tags,picture_list,tag_list]
    #translator = Translator(from_lang='chinese', to_lang='english')

    for i in trange(len(train_dirs)):
        temp_train_image = train_dirs[i]
        with open(train_path+'/'+temp_train_image+'/profile.json','r',encoding='utf-8') as file:
            profile_json = json.load(file)
        optinal_tags = profile_json['optional_tags']
        new_optinal_tags = []
        for j in range(len(optinal_tags)):
            new_optinal_tags.append(en_ch_json[optinal_tags[j]])
            if optinal_tags[j] not in en_ch_json:
                print(optinal_tags[j])
        for pictures_path_label in profile_json['imgs_tags']:
            img_path = list(pictures_path_label.keys())
            fp = open(train_path+'/'+temp_train_image+'/'+img_path[0],'rb')
            img = Image.open(fp).convert('RGB')
            img = img.resize((50,50),Image.ANTIALIAS)
            label = pictures_path_label[img_path[0]]
            #print(en_ch_json[label])
            train_data.append([new_optinal_tags,img,en_ch_json[label]])
    shuffle_fn = np.random.RandomState(535)
    shuffle_fn.shuffle(train_data)
    new_train_data = copy.deepcopy(train_data[:int(train_valid_rate*len(train_data))])
    new_valid_data = copy.deepcopy(train_data[int(train_valid_rate*len(train_data)):])
    test_data = []#二元组，[optinal_choices,img]
    for i in trange(len(test_dirs)):
        temp_test_image = test_dirs[i]
        with open(test_path + '/' + temp_test_image + '/profile.json', 'r', encoding='utf-8') as file:
            profile_json = json.load(file)
        optinal_tags = profile_json['optional_tags']
        new_optinal_tags = []
        for j in range(len(optinal_tags)):
            new_optinal_tags.append(en_ch_json[optinal_tags[j]])
            if optinal_tags[j] not in en_ch_json:
                print(optinal_tags[j])
        for pictures_path_label in profile_json['imgs_tags']:
            img_path = list(pictures_path_label.keys())
            fp_image_path = test_path+'/'+temp_test_image+'/'+img_path[0]
            fp = open(fp_image_path, 'rb')
            img = Image.open(fp).convert('RGB')
            img = img.resize((50, 50), Image.ANTIALIAS)
            test_data.append([new_optinal_tags,img,fp_image_path])
    return train_all_json,test_all_json,new_train_data,test_data,new_valid_data,en_ch_json



def get_data_img_and_text1(img_path,type_of_size):
    train_path = img_path + '/train'
    test_path = img_path + '/test'
    en_ch_json = {}
    train_all_json_path = img_path + '/train_all.json'
    test_all_json_path = img_path + '/test_all.json'
    special_labels_cn  = ["男茶色病号服","男蓝白病号服","女粉病号服","土红","矢车菊蓝","灰色条【送腰带】","红色条【送腰带】","深杏"]
    #special_labels_en  = ["man tea color","man blue white","women pink","earthy red","Cornflower blue","Gray strip belt","Red strip belt","Deep apricot"]
    special_labels_en = ["男茶色病号服", "男蓝白病号服", "女粉病号服", "土红", "矢车菊蓝", "灰色条【送腰带】", "红色条【送腰带】", "深杏"]
    with open(train_all_json_path, 'r', encoding='utf-8') as file:
        train_all_json = json.load(file)

    with open(test_all_json_path,'r',encoding='utf-8') as file:
        test_all_json = json.load(file)

    train_dirs = os.listdir(train_path)#目录train
    test_dirs = os.listdir(test_path)#目录test
    train_data = []#三元组形式，[optinal_tags,picture_list,tag_list]
    #translator = Translator(from_lang='chinese', to_lang='english')

    for i in trange(len(train_dirs)):
        temp_train_image = train_dirs[i]
        with open(train_path+'/'+temp_train_image+'/profile.json','r',encoding='utf-8') as file:
            profile_json = json.load(file)
        optinal_tags = profile_json['optional_tags']
        new_optinal_tags = []
        for j in range(len(optinal_tags)):
            # if(len(optinal_tags[j]) >20):
            #     optinal_tags[j] = optinal_tags[j][:20]
            if optinal_tags[j] in special_labels_cn:
                new_optinal_tags.append(special_labels_en[special_labels_cn.index(optinal_tags[j])])
                if special_labels_en[special_labels_cn.index(optinal_tags[j])] not in en_ch_json:
                    #en_ch_json[special_labels_en[special_labels_cn.index(optinal_tags[j])]] = optinal_tags[j]
                    en_ch_json[special_labels_en[special_labels_cn.index(optinal_tags[j])]] = special_labels_en[special_labels_cn.index(optinal_tags[j])]
            else:
                #print(optinal_tags[j])
                #new_optinal_tags.append(translator.translate(optinal_tags[j]))
                new_optinal_tags.append(optinal_tags[j])
                #print(translator.translate(optinal_tags[j]))
                #if translator.translate(optinal_tags[j]) not in en_ch_json:
                if optinal_tags[j] not in en_ch_json:
                    #en_ch_json[translator.translate(optinal_tags[j])] = optinal_tags[j]
                    en_ch_json[optinal_tags[j]] = optinal_tags[j]
        for pictures_path_label in profile_json['imgs_tags']:
            img_path = list(pictures_path_label.keys())
            fp = open(train_path+'/'+temp_train_image+'/'+img_path[0],'rb')
            img = Image.open(fp).convert('RGB')
            img = img.resize((50,50),Image.ANTIALIAS)
            #img = Image.open(train_path+'/'+temp_train_image+'/'+img_path[0]
            #print('img:{}'.format(img))
            label = pictures_path_label[img_path[0]]
            if(len(label) > 20):
                label = label[:20]
                print(label)
            #
            #
            # if label in special_labels_cn:
            #     label = special_labels_en[special_labels_cn.index(label)]
            # else:
            #     #label = translator.translate(label)
            #     label = label
                #print(label)
            # if label not in en_ch_json:
            #     en_ch_json[label] = pictures_path_label[img_path[0]]
            #new_optinal_tags = [translator.translate(optinal_tags[j]) for j in range(len(optinal_tags))]
            #print(new_optinal_tags)
            #if(len([new_optinal_tags,img,label]) != 3):
            #print([new_optinal_tags,img,label])
            train_data.append([new_optinal_tags,img,label])
    shuffle_fn = np.random.RandomState(535)
    shuffle_fn.shuffle(train_data)
    new_train_data = copy.deepcopy(train_data[:int(train_valid_rate*len(train_data))])
    new_valid_data = copy.deepcopy(train_data[int(train_valid_rate*len(train_data)):])
    #print(len(new_train_data))
    #print(len(new_valid_data))
    test_data = []#二元组，[optinal_choices,img]
    for i in trange(len(test_dirs)):
        temp_test_image = test_dirs[i]
        with open(test_path + '/' + temp_test_image + '/profile.json', 'r', encoding='utf-8') as file:
            profile_json = json.load(file)
        optinal_tags = profile_json['optional_tags']
        new_optinal_tags = []
        for j in range(len(optinal_tags)):
            if (len(optinal_tags[j]) > 20):
                optinal_tags[j] = optinal_tags[j][:20]
            if optinal_tags[j] in special_labels_cn:
                new_optinal_tags.append(special_labels_en[special_labels_cn.index(optinal_tags[j])])
                if special_labels_en[special_labels_cn.index(optinal_tags[j])] not in en_ch_json:
                    en_ch_json[special_labels_en[special_labels_cn.index(optinal_tags[j])]] = optinal_tags[j]
            else:
                #print(optinal_tags[j])
                #new_optinal_tags.append(translator.translate(optinal_tags[j]))
                new_optinal_tags.append(optinal_tags[j])
                #print(translator.translate(optinal_tags[j]))
                #if translator.translate(optinal_tags[j]) not in en_ch_json:
                if optinal_tags[j] not in en_ch_json:
                    #en_ch_json[translator.translate(optinal_tags[j])] = optinal_tags[j]
                    en_ch_json[optinal_tags[j]] = optinal_tags[j]
        for pictures_path_label in profile_json['imgs_tags']:
            img_path = list(pictures_path_label.keys())
            fp_image_path = test_path+'/'+temp_test_image+'/'+img_path[0]
            fp = open(fp_image_path, 'rb')
            img = Image.open(fp).convert('RGB')
            img = img.resize((50, 50), Image.ANTIALIAS)
            #img = Image.open()
            #new_optinal_tags = [translator.translate(optinal_tags[j]) for j in range(len(optinal_tags))]
            # for inst in optinal_tags:
            #     if translator.translate((inst)) not in en_ch_json:
            #         en_ch_json[translator.translate((inst))] = inst
            test_data.append([new_optinal_tags,img,fp_image_path])
    #print(en_ch_json)
    return train_all_json,test_all_json,new_train_data,test_data,new_valid_data,en_ch_json


#train dataset 简单，可以直接把label认为是文本
class train_Dataset(torch.utils.data.Dataset):
    def __init__(self,train_data,transform):
        self.transform = transform
        self.train_data = train_data
    def __getitem__(self, index):
        return self.transform(self.train_data[index][1]),(self.train_data[index][2],self.train_data[index][0])
    def __len__(self):
        return len(self.train_data)


#把label当作候选
class test_Dataset(torch.utils.data.Dataset):
    def __init__(self,test_data,transform):
        self.transform = transform
        self.test_data = test_data
    def __getitem__(self, index):
        return self.transform(self.test_data[index][1]),(self.test_data[index][0],self.test_data[index][2])
    def __len__(self):
        return len(self.test_data)


if __name__ == '__main__':
    train_all_json,test_all_json,train_data,test_data = get_data_img_and_text(img_path = 'img',type_of_size = 'thumbnail')
    # print(train_data)

    mean_vals = [0.471, 0.448, 0.408]
    std_vals = [0.234, 0.239, 0.242]

    #根据不同尺寸，选择不同的训练方法
    transform = transforms.Compose(
            [transforms.Resize((50, 50)),
             transforms.ToTensor(),
             transforms.Normalize(mean_vals,std_vals),

             ]
        )
    train_dataset = train_Dataset(train_data,transform)
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               shuffle = True,
                                               batch_size = 32,
                                               num_workers = 4,
                                               drop_last = False,
                                               pin_memory = True)

    test_dataset = test_Dataset(test_data,transform)
    test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                  shuffle = True,
                                                  batch_size = 1,
                                                  num_workers = 4,
                                                  drop_last = False,
                                                  pin_memory = True
                                                  )


    # print(train_dataset)
    for img,label in enumerate(train_dataloader):
        print(img)
        print(label)













