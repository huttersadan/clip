import json

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import clip
#from translate import Translator
from torch import nn, optim
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from data_input import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


torch.cuda.manual_seed(114514)
import matplotlib.pyplot as plt
class image_caption_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.optional_tags = df['optional_tags']
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(self.images[idx])
        caption = self.caption[idx]
        optional_tags = self.optional_tags[idx]
        #print(images)
        #print(caption)
        #print(optional_tags)
        #(caption,optional_tags)
        #return images, ("红色长袖西装",['红色长袖西装', '黑色长袖西装', '米色长袖西装'])
        info = {'caption':caption,'optional_tags':optional_tags}
        #print(info)
        return images,caption

class test_image_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.img_path = df["image_path"]
        self.optional_tags = df["optional_tags"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.preprocess(self.images[idx])
        img_path = self.img_path[idx]
        optional_tags = self.optional_tags[idx]
        return images, (img_path,optional_tags)


def load_data(img_path, batch_size, preprocess):
    df = {'image': [], 'caption':[],'optional_tags':[]}
    train_all_json,test_all_json,train_data,test_data,valid_data,en_ch_json = get_data_img_and_text(img_path,type_of_size = 'thumbnail')
    for single_train_data in train_data:
        df['image'].append(single_train_data[1])
        df['caption'].append(single_train_data[2])
        df['optional_tags'].append(single_train_data[0])
    dataset = image_caption_dataset(df, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    df = {'image': [], 'caption': [], 'optional_tags': []}
    for single_valid_data in valid_data:
        df['image'].append(single_valid_data[1])
        df['caption'].append(single_valid_data[2])
        df['optional_tags'].append(single_valid_data[0])
    dataset = image_caption_dataset(df, preprocess)
    valid_dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    df = {'image': [], 'optional_tags': [],'image_path':[]}
    for single_test_data in test_data:
        df['image'].append(single_test_data[1])
        df['image_path'].append(single_test_data[2])
        df['optional_tags'].append(single_test_data[0])
    dataset = test_image_dataset(df,preprocess)
    test_dataloader = DataLoader(dataset,batch_size=1,shuffle = False)
    return train_dataloader,valid_dataloader,test_dataloader,en_ch_json,test_all_json


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def load_pretrian_model(model_path):
    model, preprocess = clip.load(model_path, device=device, jit=False)  # 训练时 jit必须设置为false
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess

def train(epoch, batch_size, learning_rate, img_path,model_path):
    # 加载模型
    model, preprocess = load_pretrian_model('ViT-B/32')

    #加载数据集
    train_dataloader,valid_dataloader,test_dataloader,en_ch_json,test_all_json = load_data(img_path, batch_size, preprocess)

    #设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.05)
    epoch_losses = []
    epoch_train_acc = []
    epoch_val_acc = []
    epoch_val_loss = []
    best_val_epoch = 0
    best_val_acc = 0
    for single_epoch in range(epoch):
        total_loss = 0.0
        print_loss = 0.0
        acc_count = 0
        train_count = 0
        batches = 0
        model.train()
        for batch in tqdm(train_dataloader):
            batches += 1
            if(batches >= 1000):
                break
            optimizer.zero_grad()
            #print(batch)
            list_image,list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images
            # = info['caption']
            #list_optional_tags = info['optional_tags']
            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)
            gt = list_txt
            #print(list_txt)
            logits_per_image, logits_per_text = model(images, texts)
            #print(logits_per_text)
            tag_idx = torch.argmax(logits_per_image, dim=-1)
            #print('tag_idx:{}'.format(tag_idx))
            y_pred = []
            # for h in range(len(tag_idx)):
            #     y_pred.append(list_optional_tags[tag_idx[h]][h])
            y_pred = list(np.array(list_txt)[tag_idx.cpu().numpy()])
            #print('y_pred:{}'.format(y_pred))
            #print('gt:{}'.format(gt))
            train_count+=len(y_pred)
            for h in range(len(y_pred)):
                if y_pred[h] == gt[h]:
                    acc_count+=1
            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                #ground_truth = torch.arange(batch_size).half().to(device)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            #print(logits_per_image)
            #反向传播
            if ground_truth.shape[0] != logits_per_text.shape[0]:
                if device == "cpu":
                    ground_truth = torch.arange(logits_per_text.shape[0]).long().to(device)
                else:
                    # ground_truth = torch.arange(batch_size).half().to(device)
                    ground_truth = torch.arange(logits_per_text.shape[0], dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            print_loss += total_loss.item()
            #print('total_loss:{}'.format(total_loss.item()))

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        print('single_epoch_loss:{}'.format(print_loss/batches))
        print('train_acc = {}'.format(acc_count/train_count))
        epoch_train_acc.append(acc_count/train_count)

        epoch_losses.append(print_loss)
        if (single_epoch %4) != 0:
            continue
        # todo valid上测试一下
        valid_batches = 0
        valid_count = 0
        valid_acc_count = 0
        print_loss = 0
        model.eval()

        for batch in tqdm(valid_dataloader):
            valid_batches += 1
            if valid_batches >= 200:
                break
            #list_image, list_txt, list_optional_tags = batch
            list_image,list_txt = batch

            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)
            gt = list_txt
            logits_per_image,logits_per_text = model(images,texts)
            tag_idx = torch.argmax(logits_per_image,dim = -1)
            #print(list_txt)
            y_pred = list(np.array(list_txt)[tag_idx.cpu().numpy()])
            valid_count += len(y_pred)
            #print('y_pred:{}'.format(y_pred))
            #print('gt:{}'.format(gt))
            for h in range(len(y_pred)):
                if y_pred[h] == gt[h]:
                    valid_acc_count += 1
            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                #ground_truth = torch.arange(batch_size).half().to(device)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            if ground_truth.shape[0] != logits_per_text.shape[0]:
                if device == "cpu":
                    ground_truth = torch.arange(logits_per_text.shape[0]).long().to(device)
                else:
                    # ground_truth = torch.arange(batch_size).half().to(device)
                    ground_truth = torch.arange(logits_per_text.shape[0], dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            print_loss += total_loss.item()
            #print('total_loss:{}'.format(total_loss.item()))
        print('valid_loss:{}'.format(print_loss / valid_batches))
        print('valid_acc = {}'.format(valid_acc_count / valid_count))
        epoch_val_acc.append(valid_acc_count / valid_count)
        epoch_val_loss.append(print_loss)
        if valid_acc_count/valid_count > best_val_acc:
            best_val_acc = valid_acc_count/valid_count
            best_val_epoch = single_epoch
            torch.save(model.state_dict(), 'model/clip_model_new_{}.pt'.format(single_epoch))
    print('best_val_acc:{}'.format(best_val_acc))
    print('best_val_epoch:{}'.format(best_val_epoch))
    #state_dict = torch.load('model/CLIP_8_bz32.pth')
    #state_dict = torch.load('model/')
    #model.load_state_dict(state_dict)
    model.eval()
    print('--------------test---------------')
    ch_en_json = {}
    for key, val in en_ch_json.items():
        ch_en_json[val] = key
    new_test_all_json = {}
    model, preprocess = load_pretrian_model('ViT-B/32')
    # 在此处修改要测试的模型
    #model.load_state_dict(torch.load('model/clip_model.pt'))
    #model = model.to(device)
    if epoch != 0:
        model_path = 'model/clip_model_new_'+str(best_val_epoch)+'.pt'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    for batch in tqdm(test_dataloader):

        list_image,(image_path,li_optional_tags) = batch
        image_path = image_path[0]
        list_optional_tags = []
        #for k in range(len(li_optional_tags)):
            # if len(li_optional_tags[k][0]) > 20:
            #     list_optional_tags.append(li_optional_tags[k][0][:20])
            # else:
        #    list_optional_tags.append(li_optional_tags[k][0])
        list_optional_tags = [li_optional_tags[i][0] for i in range(len(li_optional_tags))]
        #print(list_optional_tags)
        #print(image_path)
        list_image = list_image.to(device)
        texts = clip.tokenize(list_optional_tags).to(device)
        with torch.no_grad():
            image_features = model.encode_image(list_image)
            text_features = model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values,indices = similarity[0].topk(1)
            y_pred_label = list_optional_tags[indices]
            #print('y_pred_label:{}'.format(y_pred_label))
            # if(len(y_pred_label) >= 20):
            #     print('y_pred_label:{}'.format(y_pred_label))
            logits_per_image, logits_per_text = model(list_image, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        np_probs = np.array(probs[0])
        #print(list_optional_tags[np.argmax(np_probs)])
        text_img_split = image_path.split('/')
        text_img_path = text_img_split[-1]
        profile_path_total = text_img_split[:-1]
        profile_path = profile_path_total[0]
        for temp_path in profile_path_total[1:]:
            profile_path = profile_path + '/' + temp_path
        profile_path += '/profile.json'
        with open(profile_path, 'r', encoding='utf-8') as file:
            params = json.load(file)
            # print(text_img_path)
            for para_idx in range(len(params['imgs_tags'])):
                img_path = list(params['imgs_tags'][para_idx].keys())
                if img_path[0] == text_img_path:
                    params['imgs_tags'][para_idx][img_path[0]] = ch_en_json[list_optional_tags[np.argmax(np_probs)]]
                    break
        file.close()
        with open(profile_path, 'w', encoding='utf-8') as file:
            json.dump(params, file, ensure_ascii=False)
        file.close()
        new_test_all_json[text_img_split[2]] = params
    with open('test_all.json','w',encoding = 'utf-8') as file:
        json.dump(new_test_all_json,file,ensure_ascii=False)
    file.close()
    plt.figure('train_losses')
    plt.title('train_losses')
    epoches = [k for k in range(len(epoch_losses))]
    plt.plot(epoches,epoch_losses)
    plt.savefig("train_losses.pdf")

    plt.figure('val_losses')
    plt.title('val_losses')
    epoches = [k for k in range(len(epoch_val_loss))]
    plt.plot(epoches, epoch_val_loss)
    plt.savefig("val_losses.pdf")
    plt.figure('val_acc')
    plt.title('val_acc')
    epoches = [k for k in range(len(epoch_val_acc))]
    plt.plot(epoches, epoch_val_acc)
    plt.savefig("val_acc.pdf")

    plt.figure('train_acc')
    plt.title('train_acc')
    epoches = [k for k in range(len(epoch_train_acc))]
    plt.plot(epoches, epoch_train_acc)
    plt.savefig("train_acc.pdf")

def main():
    epoch = 80
    batch_size = 3
    learning_rate = 5e-7
    img_path = 'thumbnail'
    model_path = 'model/clip_model_new_56.pt'
    train(epoch, batch_size, learning_rate, img_path,model_path)

if __name__ == '__main__':
    main()
