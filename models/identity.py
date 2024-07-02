"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2023/5/14 16:07
@Software: PyCharm 
@File : identity.py
"""
import torch
import torch.nn.functional as F
from backbone import facenet,irse,ir152
# 使用生成id用下面的
#from backbone import facenet,irse,ir152

from tqdm import tqdm

import cv2
import numpy as np
import os

class TestFace():
    def __init__(self):
        super(TestFace, self).__init__()
        self.gpu = True
        self.device = torch.device('cuda')if self.gpu else  torch.device('cpu')
        self.facemodels()

    def facemodels(self):
        self.targe_models = {}
        #selected=['ir152','irse50','facenet','mobile_face']
        selected=['facenet']
        for model in selected:
            if model == 'ir152':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = ir152.IR_152((112, 112))
                if self.gpu:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/ir152.pth'))
                else:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/ir152.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'irse50':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = irse.Backbone(50, 0.6, 'ir_se')
                if self.gpu:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/irse50.pth'))
                else:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/irse50.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'facenet':
                self.targe_models[model] = []
                self.targe_models[model].append((160, 160))
                fr_model = facenet.InceptionResnetV1(num_classes=8631, device=self.device)
                if self.gpu:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/facenet.pth'))

                else:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/facenet.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)
            if model == 'mobile_face':
                self.targe_models[model] = []
                self.targe_models[model].append((112, 112))
                fr_model = irse.MobileFaceNet(512)
                if self.gpu:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/mobile_face.pth'))

                else:
                    fr_model.load_state_dict(torch.load('/home/jijunhao/DMDP/resources/mobile_face.pth',map_location='cpu'))
                fr_model.to(self.device)
                fr_model.eval()
                self.targe_models[model].append(fr_model)

    def test_verification(self, img_a, img_b):

        self.facemodels()
        targeted_loss_list = []

        for model_name in self.targe_models.keys():
            target_loss_A = self.cal_loss(img_a, img_b, model_name, self.targe_models)
            targeted_loss_list.append(target_loss_A)

        targeted_loss_list=torch.stack(targeted_loss_list)

        return targeted_loss_list

    def cos_simi(self, emb_1, emb_2):
        sims=torch.sum(torch.mul(emb_2, emb_1), dim=1) / emb_2.norm(dim=1) / emb_1.norm(dim=1)
        return torch.mean((sims))

    def cal_loss(self, source, target, model_name, target_models):  # 计算损失
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        with torch.no_grad():
            source_resize = F.interpolate(source, size=input_size, mode='bilinear')
            target_resize = F.interpolate(target, size=input_size, mode='bilinear')
            emb_source = fr_model(source_resize)
            emb_target = fr_model(target_resize).detach()
            #cos_loss = self.cos_simi(emb_source, emb_target)
            cos_loss = torch.cosine_similarity(emb_source, emb_target)
        return torch.mean(cos_loss)

    def pred_id(self, source, model_name,target_models):
        input_size = target_models[model_name][0]
        fr_model = target_models[model_name][1]
        with torch.no_grad():
            source_resize = F.interpolate(source, size=input_size, mode='bilinear')
            emb_source = fr_model(source_resize)
        return emb_source

if __name__ == '__main__':
    TestFace = TestFace()

    ori_path = "/home/jijunhao/DPINDEX/dataset/img"

    pro_path = "/home/jijunhao/DPINDEX/output/helen/our_block_8_epsilon_01"
    files = os.listdir(pro_path)
    print(len(files))


    dataid = []
    for file_name in tqdm(files):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            data = cv2.imread(os.path.join(ori_path, file_name))
            data = np.array(data, dtype=np.float32) / 255.0
            data = np.transpose(data, (2, 0, 1))
            data = torch.from_numpy(data).unsqueeze(0).to('cuda')
            # 保存向量id
            id = TestFace.pred_id(data, 'facenet', TestFace.targe_models)

            data2 = cv2.imread(os.path.join(pro_path, file_name.split('.')[0] + '.jpg'))
            data2 = np.array(data2, dtype=np.float32) / 255.0
            data2 = np.transpose(data2, (2, 0, 1))
            data2 = torch.from_numpy(data2).unsqueeze(0).to('cuda')
            # 保存向量id
            id2 = TestFace.pred_id(data2, 'facenet', TestFace.targe_models)
            dataid.append(TestFace.cos_simi(id2, id))
    filtered_dataid = [value for value in dataid if value < 0.4]

    print(1-len(filtered_dataid) / len(dataid))
    print(min(dataid), max(dataid))
    print(sum(dataid) / len(dataid))
    #
    # TestFace = TestFace()
    #
    # folder_path = "/home/jijunhao/DMDP/synthesis/dataset/10.0"
    # files = os.listdir(folder_path)
    #
    # dataid = []
    # for file_name in tqdm(files[:500]):
    #     orignal = cv2.imread('/home/jijunhao/diffusion/data/img_align_celeba'+"/"+file_name.replace('.png','.jpg'))
    #     # orignal = cv2.imread('/home/jijunhao/diffusion/data/img_align_celeba/000001.jpg')
    #     orignal = np.array(orignal, dtype=np.float32) / 255.0
    #     orignal = np.transpose(orignal, (2, 0, 1))
    #     orignal = torch.from_numpy(orignal).unsqueeze(0).to('cuda')
    #     orignal_id = TestFace.pred_id(orignal, 'facenet', TestFace.targe_models)
    #
    #     data = cv2.imread(os.path.join(folder_path, file_name))
    #     data = np.array(data, dtype=np.float32) / 255.0
    #     data = np.transpose(data, (2, 0, 1))
    #     data = torch.from_numpy(data).unsqueeze(0).to('cuda')
    #     # 保存向量id
    #     id = TestFace.pred_id(data, 'facenet', TestFace.targe_models)
    #     dataid.append(TestFace.cos_simi(orignal_id, id))
    #
    # print(min(dataid),max(dataid))
    # print(sum(dataid)/len(dataid))

    # folder_path = "/home/jijunhao/diffusion/data/img_align_celeba/"
    # files = os.listdir(folder_path)
    #
    # for file_name in tqdm(files):
    #     if file_name.endswith(".jpg"):
    #         data = cv2.imread(os.path.join(folder_path, file_name))
    #         data = np.array(data, dtype=np.float32) / 255.0
    #         data = np.transpose(data, (2, 0, 1))
    #         data = torch.from_numpy(data).unsqueeze(0).to('cuda')
    #         # 保存向量id
    #         id = TestFace.pred_id(data, 'ir152', TestFace.targe_models)
    #         with torch.no_grad():
    #             torch.save(id.to("cpu"), os.path.join('datasets/id/', file_name.replace('.jpg', '.pt')))
    # data = cv2.imread("./datasets/image/image_512_downsampled_from_hq_1024/"+str(0)+".jpg")
    # data = np.array(data, dtype=np.float32) / 255.0
    # data = np.transpose(data, (2, 0, 1))
    # data = torch.from_numpy(data).unsqueeze(0).to('cuda')
    # data1 = cv2.imread("./datasets/image/image_512_downsampled_from_hq_1024/"+str(3)+".jpg")
    # data1 = np.array(data1, dtype=np.float32) / 255.0
    # data1 = np.transpose(data1, (2, 0, 1))
    # data1 = torch.from_numpy(data1).unsqueeze(0).to('cuda')
    # print(TestFace.test_verification(data,data1))



