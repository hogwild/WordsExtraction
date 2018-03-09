# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:48:59 2017

@author: Gu
"""
#from PIL import Image
#import pytesseract
import numpy as np
import cv2
import pickle
import nms
#from pywt import WaveletPacket2D
#from sklearn.neighbors import KNeighborsClassifier



 
'''Region selection'''
def onmouse1(event,x,y,flags,param):
    global drag_start, sel
    workimg = np.copy(frame)
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0,0,0,0
    elif drag_start:
        if flags and cv2.EVENT_FLAG_LBUTTON:           
            minpos = min(drag_start[0],x),min(drag_start[1],y)
            maxpos = max(drag_start[0],x),max(drag_start[1],y)
            sel = minpos[0],minpos[1],maxpos[0],maxpos[1]           
            cv2.rectangle(workimg,(sel[0],sel[1]),(sel[2],sel[3]),(0,0,255),2)
            cv2.imshow("frame",workimg)
        else:            
            print("Selection is completed.")
            drag_start = None


def onmouse2(event, x, y, flags, param):
    global drag_start, sel
    workimg = np.copy(frame)
#    print("event1:", event)
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = x, y
        sel = 0, 0, 0, 0
    elif drag_start:
        if  flags == 33:
            minpos = min(drag_start[0],x),min(drag_start[1],y)
            maxpos = max(drag_start[0],x),max(drag_start[1],y)
            sel = minpos[0],minpos[1],maxpos[0],maxpos[1]           
            cv2.rectangle(workimg,(sel[0],sel[1]),(sel[2],sel[3]),(0,0,255),2)
            cv2.imshow("frame",workimg)
        else: 
            print("Selection is completed.")
            drag_start = None
    

            
#def onmouse3(event,x,y,flags,param):
#    global positive_position, negative_position 
#    key = cv2.waitKey(100)&0xFF
#    if key == ord('p'):
#        if event == cv2.EVENT_LBUTTONDOWN:
#            positive_position = y, x  #the y is the raw number;x is the colom number
#            cv2.circle(pre_seg_img,(x, y), 2, (0, 0, 1), 2)
#            pos_sample_pos.append(positive_position)
#            print("Positive Sample selected.", x, y)
#    if key == ord('n'):
#        if event == cv2.EVENT_LBUTTONDOWN:
#            negative_position = y, x  
#            cv2.circle(pre_seg_img,(x, y), 2,(0, 0, 0), 2)
#            neg_sample_pos.append(negative_position)
#            print("Negative Sample selected.", x, y)

    
## load classifier
knn = pickle.load(open('./classifier_knn.pk', 'rb'))
svm = pickle.load(open('./classifier_svm.pk', 'rb'))

## load the music video
cap = cv2.VideoCapture('../samples/lishuangjiang.mkv')
sel = (0,0,0,0)
drag_start = None
pos_sample_pos = []
neg_sample_pos = []

count = 0
#c1 = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 8))
#c2 = cv2.getStructuringElement(cv2.MORPH_RECT,(16, 14))
ret, frame = cap.read()
Y, X, Z = frame.shape
LINE_GAP = 10
region = (0, 0, 0, 0)
one_char = (0, 0, 0, 0)
five_chars = (0, 0, 0, 0)
samples = []
SIZE = (16, 16)
c1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
c2 = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 6))
mser = cv2.MSER_create()
while(cap.isOpened()):
    ret, frame = cap.read()
    try: ## avoid the bad ending of .mkv files
#        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
    except:
           break 
    if cv2.waitKey(10) & 0xFF == ord('q'): ## quit  
        break
    #### select the region of the subtitles:
    if cv2.waitKey(1) & 0xFF == ord('s'): ## select the region of the subtitles
        k = cv2.waitKey(1) & 0xFF
        cv2.setMouseCallback('frame', onmouse2, frame)
        while k != ord('d'):          ## when the region selection is done, press "d" to go on.
            k = cv2.waitKey(1) & 0xFF
        region = sel[:]
        print("The region is: ", region)
#        while k != ord('f'):
#            k = cv2.waitKey(1) & 0xFF
#            five_chars = sel[:]
##            print("the threshold is", frame[sel[1], sel[0],:])
#            if count < 1:
#               threshold = np.copy(frame[sel[1], sel[0],:])
#        print("The size of one character is: {}".format(five_chars))
        while k != ord('g'):
            k = cv2.waitKey(1) & 0xFF
            one_char = sel[:]
        print("The size of gap is: {}".format(one_char))
   
    if sum(region) > 0:    
        count += 1
#        if count < 2:
#            print("the threshold is", threshold)
            
        #### get the threshold for binary image
        GAP = int(one_char[2] - one_char[0])
        HEIGHT = int(one_char[3] - one_char[1])
#        block = np.copy(frame[one_char[1]:one_char[3], one_char[0]:one_char[2]])
#        block_gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
#        block_gray = cv2.adaptiveThreshold(block_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
#        img, contours_block, hierarchy = cv2.findContours(block_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        GAP = int((five_chars[2] - five_chars[0] - one_char[2] + one_char[0])/4)
#        print("the gap is:", GAP)
#        for i, cnt in enumerate(contours_block):
#            x, y, w, h = cv2.boundingRect(cnt)
#            cv2.rectangle(block, (x, y), (x+w, y+h), (0, 0, 255), 2)
        LEFT_BOUND = region[0]
        RIGHT_BOUND = region[2]
        UP_BOUND = region[1]
        LOW_BOUND = region[3]
#        CHAR_WIDTH = GAP
        MID = int((LOW_BOUND - UP_BOUND)/2)
        img_title = np.copy(frame[UP_BOUND:LOW_BOUND, LEFT_BOUND:RIGHT_BOUND, :])
#        img_title = cv2.resize(img_title, None, fx=0.5, fy=0.5) 
        gray = cv2.cvtColor(img_title, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5)
        regions, boxes = mser.detectRegions(gray)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        keep = []
        for c in hulls:
            x, y, w, h = cv2.boundingRect(c)
            if w*h <= GAP * HEIGHT and w < GAP * 1.1 and h < HEIGHT * 1.1:
                keep.append([x, y, x + w, y + h])
        keep2 = np.array(keep)
        pick = nms.nms(keep2, 0.1)
        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(img_title, (startX, startY), (endX, endY), (0, 0, 255), 2)
#        sobel = cv2.Sobel(gray, cv2.CV_16S, 1, 0, 3)
#        sobel = cv2.convertScaleAbs(sobel)
#        ret, gray = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
#        cv2.imwrite('./data/title_example/test.png', img_title)
        
#        gray = cv2.GaussianBlur(gray, (3, 3), 0)
#        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5)
#        gray = cv2.dilate(gray, c2, iterations=1)
#        gray = cv2.erode(gray, c1, iterations=1)
#        
#        img, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##        for i, cnt, in enumerate(contours):
##            x, y, w, h = cv2.boundingRect(cnt)
##            cv2.rectangle(img_title, (x, y), (x+w, y+h), (0, 0, 255), 2)
#        
#        new_contours = []
#        for i, cnt in enumerate(contours):
#            x, y, w, h = cv2.boundingRect(cnt)
#            C_y = int(y+(h/2))
#            if  0.4*MID < C_y < 0.6*MID or 1.4*MID < C_y < 1.6*MID:
#                if  0.3*GAP < w < 1.05*GAP and 0.2*MID < h < MID:
#                    new_contours.append((x, y, w, h))
#                    cv2.rectangle(img_title, (x, y), (x+w, y+h), (0, 0, 255), 2)
#        new_contours.sort(key=lambda x:x[0])
#        for i, cnt in enumerate(new_contours):
#            j = i + 1
#            if j == len(new_contours):
#                break
#            while j < len(new_contours) and abs(new_contours[j][1] - cnt[1]) < 0.2*MID:
#                j += 1
        cv2.imshow('titles', gray)    
#        cv2.imshow("titles3", block_gray)
        cv2.imshow('titles2', img_title)
        
        
        
     
'''                                

    ### draw the white lines
        i = 1
        while i*CHAR_WIDTH <= RIGHT_BOUND - LEFT_BOUND:
            start = (i-1) * CHAR_WIDTH
            end = i * CHAR_WIDTH
            img_up = cv2.resize(gray[:MID, start:end], SIZE)
            
#            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            img_up = cv2.GaussianBlur(img_up, (3, 3), 0)
            img_up = cv2.adaptiveThreshold(img_up, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
            wp2 = WaveletPacket2D(img_up, 'db2', 'symmetric', maxlevel=2)
            f1 = wp2['a'].data
            f2 = wp2['h'].data
            f3 = wp2['d'].data
            f4 = wp2['v'].data
            f = np.concatenate((f1.reshape(-1), f2.reshape(-1), f3.reshape(-1), f4.reshape(-1)))
            label = knn.predict(f.reshape(-1, f.size))
            img_title[:, end, 0] = 255
            img_title[:, end, 1] = 0
            img_title[:, end, 2] = 0
            
            if label == 1:
                new_contours.append((start, 0, CHAR_WIDTH, MID))
#                img_title[:int(MID/2), start:end, 0] = 0
#                img_title[:int(MID/2), start:end, 1] = 0
#                img_title[:int(MID/2), start:end, 2] = 255
            
            img_low = cv2.resize(gray[MID:, start:end], SIZE)
            img_low = cv2.adaptiveThreshold(img_low, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
#            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            img_low = cv2.GaussianBlur(img_low, (3, 3), 0)
            wp2 = WaveletPacket2D(img_low, 'db2', 'symmetric', maxlevel=2)
            f1 = wp2['a'].data
            f2 = wp2['h'].data
            f3 = wp2['d'].data
            f4 = wp2['v'].data
            f = np.concatenate((f1.reshape(-1), f2.reshape(-1), f3.reshape(-1), f4.reshape(-1)))
            label = knn.predict(f.reshape(-1, f.size))
            img_title[:, end, 0] = 255
            img_title[:, end, 1] = 0
            img_title[:, end, 2] = 0
            
            if label == 1:
                new_contours.append((start, MID, CHAR_WIDTH, MID))
#                img_title[MID+int(MID/2):, start:end, 0] = 0
#                img_title[MID+int(MID/2):, start:end, 1] = 0
#                img_title[MID+int(MID/2):, start:end, 2] = 255
#                img_title[:MID, end, 0] = 0
#                img_title[:MID, end, 1] = 0
#                img_title[:MID, end, 2] = 255
                
#            blocks.append(gray[:MID, start:end])
#            cv2.imwrite("./titles/{}_{}_up.png".format(count, i), img_title[:MID, start:end])
##            cv2.imshow('chinese-char_up', img_title[:MID, start:end])
#            cv2.imwrite("./titles/{}_{}_low.png".format(count, i), img_title[MID:, start:end])
#            cv2.imshow('chinese-char_low', img_title[MID:, start:end])
#            gray[:, i*CHAR_WIDTH] = 255
            for cnt in new_contours:
                x, y, w, h = cnt
                cv2.rectangle(img_title, (x, y), (x+w, y+h), (0, 0, 255), 2)
            i += 1
          
        cv2.imshow('titles', img_title)
#    if count > 1000:
#        pickle.dump(samples, open('./titles/samples.pk', 'w'))
#        break
        

        
#        cv2.imwrite("./titles/{}.png".format(count), gray)
        
    #### detect the rhythm:
'''      
 
    

        
cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
#cv2.waitKey()
 
#cv2.namedWindow('img', 1)
#cv2.setMouseCallback('img', onmouse1, frame) 
#sel = (0,0,0,0)
#drag_start = None
#cv2.imshow("img",frame)
#k = cv2.waitKey(20)&0xFF


#while True:
#    cv2.imshow("img",frame)
#    k = cv2.waitKey(20)&0xFF
#    if k == ord('b'):
#        break
#    else:#if k == ord('a'):
#        cv2.rectangle(frame,(sel[0],sel[1]),(sel[2],sel[3]),(0,0,255),1)
#        cv2.imshow("img",frame)    
#        cv2.destroyAllWindows()        
#print("The region selected is: ",sel)
#for imgName in filename_list:
#    image = path+imgName
#    if j==0:
#        img = cv.imread(image)
#        img = cv.resize(img,(1200,900))
#        set_region = raw_input('Would you like to set the region by hand, yes or no ?  ')
#        if set_re gion.lower() in ['y','Y','yes','Yes','YES']:
#            sel = range(4)
#            sel[0] = int(input('Please input x1:   '))
#            sel[1] = int(input('Please input y1:   '))
#            sel[2] = int(input('Please input x2:   '))
#            sel[3] = int(input('Please input y2:   '))            
#        else:
#            print("Press 'b' when the region selection is completed.")
#            cv2.namedWindow("img",1)
#            cv2.setMouseCallback("img", onmouse1,img)
#            sel = (0,0,0,0)
#            drag_start = None
#            is_img = True
#            while (1):
#                if is_img:
#                    cv2.imshow("img",img)
#                else:
#                    cv2.imshow("img",img2)
#                k = cv2.waitKey(20)&0xFF
#                if k == ord('b'):
#                    break
#                else:#if k == ord('a'):
#                    is_img = False
#                    img2 = np.copy(img)
#                    cv2.rectangle(img2,(sel[0],sel[1]),(sel[2],sel[3]),(0,0,255),2)
#                    cv2.imshow("img",img2)                
#            cv2.destroyAllWindows()        
#        print("The region selected is: ",sel)
##        patch = img[sel[1]:sel[3],sel[0]:sel[2]]
##        centroid = np.array(((sel[3]-sel[1])/2.0,(sel[2]-sel[0])/2.0))
##        print "The centroid is: ",centroid
#        '''take the color velue of the region center'''
#        b[j]=img[sel[1]+(sel[3]-sel[1])/2,sel[0]+(sel[2]-sel[0])/2,0]
#        g[j]=img[sel[1]+(sel[3]-sel[1])/2,sel[0]+(sel[2]-sel[0])/2,1]
#        r[j]=img[sel[1]+(sel[3]-sel[1])/2,sel[0]+(sel[2]-sel[0])/2,2]
#    else:
#        img = cv2.imread(image)
#        img = cv2.resize(img,(1200,900))
#        b[j]=img[sel[1]+(sel[3]-sel[1])/2,sel[0]+(sel[2]-sel[0])/2,0]
#        g[j]=img[sel[1]+(sel[3]-sel[1])/2,sel[0]+(sel[2]-sel[0])/2,1]
#        r[j]=img[sel[1]+(sel[3]-sel[1])/2,sel[0]+(sel[2]-sel[0])/2,2]
#    j+=1


 

#while(cap.isOpened()):
#    ret, frame = cap.read()     
#    cv2.imshow('frame', frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#        


