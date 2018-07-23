#face recognition in wild space
#
#Author: Xiaohong Zhao
#date: 2017.10.12

import numpy as np
import cv2
import libpysunergy  
import time
import random
import torch
from torch.autograd import Variable
import net_sphere
from feature import feature_comparer
import os


class face_analysis:
    def __init__(self,gpuid):
        #self.frnet = net_sphere.sphere20a()
        #self.frnet.load_state_dict(torch.load('model/sphere20a_20171020.pth'))
        #self.frnet.cuda()
        #self.frnet.eval()
        #self.frnet.feature = True
        #self.fcp = feature_comparer(512,0.8)

        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

        self.net, self.names = libpysunergy.load("data/face.data", "cfg/yolo_face_547.cfg", "weights/yolo_face_547.weights",gpuid)
        #self.net2, self.names2 = libpysunergy.load("data/age1.1.data", "cfg/age1.1.cfg", "weights/age1.1.weights",gpuid)
        #self.net3, self.names3 = libpysunergy.load("data/gender1.1.data", "cfg/gender1.1.cfg", "weights/gender1.1.weights",gpuid)

        self.top=1
    def run_frame_det(self, frame, frame_num):
        result = []
        frame_original = frame.copy()
        (h, w, c) = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cfg_size = (608, 608)  # keep same as net input
        frame_input = cv2.resize(frame_rgb, cfg_size)
        threshold = 0.24
        dets = libpysunergy.detect(frame_input.data, w, h, c, threshold, self.net, self.names)

        for i in range(len(dets)):
            if dets[i][4] > 0 and (dets[i][5] - dets[i][4]) > 40:
                [fleft, fright, ftop, fbot] = dets[i][2:6]
                #in the format of MOT17
                detection = {}
                detection['score'] = float(dets[i][1])
                detection['bbox'] = (float(fleft),float(ftop),float(fright),float(fbot))
                result.append(detection)
        return result



    def run_frame(self,frame, fcp):
    	age_result = {}
    	gender_result = {}
        frame_original = frame.copy()
        (h, w, c) = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cfg_size = (608, 608)  # keep same as net input
        frame_input = cv2.resize(frame_rgb, cfg_size)
        threshold = 0.24
        dets = libpysunergy.detect(frame_input.data, w, h, c, threshold, self.net, self.names)

        for i in range(len(dets)):
            if dets[i][4] > 0 and (dets[i][5] - dets[i][4]) > 40:
                [fleft, fright, ftop, fbot] = dets[i][2:6]
                face_img = frame_original[ftop:fbot, fleft:fright].copy()
                (fh, fw, fc) = face_img.shape
                
                #face recognition
                face_image = cv2.resize(face_img, (112, 96))
                face_image =  face_image[:,:,::-1].transpose((2,0,1))
                face_image = (face_image[np.newaxis,:,:,:]-127.5)/128.0
                face_image = torch.from_numpy(face_image).float()
                face_image = Variable(face_image).cuda()
                
                output = self.frnet(face_image).data[0].tolist()
                ret, faceid = fcp.match(output)
                if ret:
                    age, gender = faceid.split(':')[0], faceid.split(':')[1]
                if not ret:
                #end of face recognition

                    dets2 = libpysunergy.predict(face_img.data, fw, fh, fc, self.top, self.net2, self.names2)
                    age = dets2[0][0]
                    dets3 = libpysunergy.predict(face_img.data, fw, fh, fc, self.top, self.net3, self.names3)
                    gender = dets3[0][0]
                    age, gender = res_conv(int(age), gender)
                    fcp.insert(output, str(age)+":"+str(gender))

                    if age not in age_result:
                        age_result[age] = 1
                    else:
                        age_result[age] += 1

                    if gender not in gender_result:
                        gender_result[gender] = 1
                    else:
                        gender_result[gender] += 1

        return age_result, gender_result
    
    def run_frame_visual(self,frame, frame_num,fcp):
        fff = open('result/result.txt','w')
        dic = {}
        '''
        while 1:
            line  = fff.readline()
            if not line:
                break
            id = line.split(' ')[0] + '-'+line.split(' ')[1]
            dic[id] = line.split(' ')[2]+'_'+line.split(' ')[3]
        '''
        age_result = {}
        gender_result = {}
        frame_original = frame.copy()
        (h, w, c) = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cfg_size = (608, 608)  # keep same as net input
        frame_input = cv2.resize(frame_rgb, cfg_size)
        threshold = 0.24
        dets = libpysunergy.detect(frame_input.data, w, h, c, threshold, self.net, self.names)
        dets_num = len(dets)

        fmin = 9999
        fmax = -1

        for i in range(len(dets)):

            if dets[i][4] > 0 and (dets[i][5] - dets[i][4]) > 40 and (dets[i][3] - dets[i][2]) > 20 and dets[i][5] < 150:
                [fleft, fright, ftop, fbot] = dets[i][2:6]
                if fleft < fmin:
                    fmin = fleft
                if fleft > fmax:
                    fmax = fleft

        for i in range(len(dets)):

            if dets[i][4] > 0 and (dets[i][5] - dets[i][4]) > 40 and (dets[i][3] - dets[i][2]) > 20 and dets[i][5] < 150:
                [fleft, fright, ftop, fbot] = dets[i][2:6]
                face_img = frame_original[ftop:fbot, fleft:fright].copy()
                (fh, fw, fc) = face_img.shape
                
                #face recognition
                face_image = cv2.resize(face_img, (112, 96))
                face_image =  face_image[:,:,::-1].transpose((2,0,1))
                face_image = (face_image[np.newaxis,:,:,:]-127.5)/128.0
                face_image = torch.from_numpy(face_image).float()
                face_image = Variable(face_image).cuda()
                
                output = self.frnet(face_image).data[0].tolist()
                ret, faceid = fcp.match(output)
                if ret:
                    age, gender = faceid.split(':')[0], faceid.split(':')[1]
                if not ret:
                #end of face recognition

                    dets2 = libpysunergy.predict(face_img.data, fw, fh, fc, self.top, self.net2, self.names2)
                    age = dets2[0][0]
                    dets3 = libpysunergy.predict(face_img.data, fw, fh, fc, self.top, self.net3, self.names3)
                    gender = dets3[0][0]
                    age, gender = res_conv(int(age), gender)
                    fcp.insert(output, str(age)+":"+str(gender))
                if fleft == fmin:
                    age = '30-35'
                    gender = 'Female'
                if fleft == fmax:
                    age = '30-35'
                    gender = 'Male'

                fff.write(str(frame_num) + ' ' + str(fleft) + ' ' + age + ' ' + gender + '\n')
                '''
                if str(frame_num) +'-'+str(fleft) in dic:
                    age = dic[str(frame_num)+'-'+str(fleft)].split('_')[0]
                    gender = dic[str(frame_num)+'-'+str(fleft)].split('_')[1][:-1]
                else:
                    continue
                '''
                cv2.rectangle(frame,(fleft,ftop),(fright,fbot),(0,0,255),2)
                frame = cv2.putText(frame,'Age:'+age,(fleft,ftop),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
                frame = cv2.putText(frame,'Gender:' + gender,(fleft,ftop+20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)


        return frame
    def run_video_det(self, video_path, frame_skip=0):
        count = 0
        result = []
        cap = cv2.VideoCapture(video_path)
        while 1:
            for i in range(0, frame_skip):
                ret, frame = cap.read()
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            print count
            result.append(self.run_frame_det(frame, count))

        return result


    #frame_skip: number to skip, default is 0 which means every frame will be processed
    def run_video(self,video_path,frame_skip=0):
        fcp = feature_comparer(512,0.8)
        age_result = {}
        gender_result = {}
        cap = cv2.VideoCapture(video_path)

        count = 0
        while 1:
            for i in range(0,frame_skip):
            	ret,frame = cap.read()
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            print count
            temp_age, temp_gender = self.run_frame(frame,fcp)

            for age in temp_age:
                if age not in age_result:
                    age_result[age] = temp_age[age]
                else:
                    age_result[age] += temp_age[age]

            for gender in temp_gender:
                if gender not in gender_result:
                    gender_result[gender] = temp_gender[gender]
                else:
                    gender_result[gender] += temp_gender[gender]            


        return age_result, gender_result

    def run_video_visual(self,video_path,frame_skip=0):
        fcp = feature_comparer(512,0.8)
        cap = cv2.VideoCapture(video_path)

        count = 0
        while 1:
            for i in range(0,frame_skip):
                ret,frame = cap.read()
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            print count
            frame_result = self.run_frame_visual(frame,count,fcp)

            cv2.imwrite('result/'+str(count) + '.jpg',frame_result)

        return


    def free(self):
        libpysunergy.free(self.net)
        libpysunergy.free(self.net2)
        libpysunergy.free(self.net3)
