from PIL import Image
#import matplotlib.image as img
from os import listdir
import random
import numpy as np
import math
from natsort import natsorted


class preprocess(object):

    def __init__(self, dataset_location="", batch_size=1,shuffle=False):
        self.location=dataset_location
        self.num=batch_size
        self.shuffle=shuffle

        self.list=listdir(self.location)
        #self.list=natsorted(list1,key=lambda s:s.lower())
        #print("list1",list1)
        #print("sorted list",self.list)
        if self.shuffle==True:
            self.flist=random.sample(self.list,self.num)
        else:
            self.flist=self.list[:self.num]
        for i in range(len(self.flist)):
            self.flist[i]=self.location+self.flist[i]

        #print(self.flist)
        #print(self.flist)
        self.idx=[]

        for img in self.flist:
              try:
                #print(img)
                s=img.split('_')
                self.idx.append(int(s[0]))
              except Exception:
                self.idx.append('NIL')
        self.length=len(self.list)
        self.seed=None


    def rescale(self,s):
        dict={}
        for oimg in self.flist:

            oarray=Image.open(oimg)
            oarray=np.array(oarray)
            oshape=oarray.shape
            #print("oarraydtype",oarray.dtype)
            #print("Originalshape",oshape)
            #print("Originalarray",oarray[0][0])


            oimg_h,oimg_w=oarray.shape[:2]
            nimg_h,nimg_w=oarray.shape[0]*s,oarray.shape[1]*s
            x_ratio=float((oimg_w-1)/nimg_w)
            y_ratio=float((oimg_h-1)/nimg_h)

            value=[0,0,0]
            if len(oarray.shape)>2:
                narray=np.zeros((oshape[0]*s,oshape[1]*s,3),np.uint8)
            else:
                narray = np.zeros((oshape[0]*s,oshape[1]*s),np.uint8)
            #print(narray.shape)
            for i in range(nimg_h):
                for j in range(nimg_w):
                    x1=math.floor(x_ratio*j)
                    y1=math.floor(y_ratio*i)
                    x2=math.ceil(x_ratio*j)
                    y2=math.ceil(y_ratio*i)
                    #print(x1,y1,x2,y2)
                    xw=(x_ratio*j)-x1
                    yw=(y_ratio*i)-y1
                    #print(i,j)
                    a = oarray[y1, x1]
                    #print(a)
                    b = oarray[y1, x2]
                    c = oarray[y2, x1]
                    d = oarray[y2, x2]
                    if len(oarray.shape)>2:
                        for k in range(3):
                            value[k]=((a[k]) * (1 - xw) * (1 - yw)) + (b[k]) * xw * (1 - yw) + (c[k]) * yw * (1 - xw) + (d[k]) * xw * yw
                        #p = ((a) * (1 - xw) * (1 - yw)) + (b) * xw * (1 - yw) + (c) * yw * (1 - xw) + (d) * xw * yw
                        #print(value)
                        p=np.array([value[0],value[1],value[2]])
                        #print(p)
                        narray[i,j] = p

                    else:
                        p = ((a) * (1 - xw) * (1 - yw)) + (b) * xw * (1 - yw) + (c) * yw * (1 - xw) + (d) * xw * yw
                        narray[i][j]=p

            dict[oimg]=narray
            #print(narray.shape)
            #nimg=Image.fromarray(narray)
            #Image._show(nimg)
        return dict

    def resize(self, h, w):
        dict={}

        for oimg in self.flist:
            oarray=Image.open(oimg)
            #Image._show(oarray)
            oarray=np.array(oarray)
            #print(oarray)
            #print(oarray.shape)
            if len(oarray.shape)>2:
                narray=np.zeros((h,w,3),np.uint32)
            else:
                narray=np.zeros((h,w),np.uint32)
            #print(narray.shape)
            yScale=h/(oarray.shape[0])
            xScale=w/(oarray.shape[1])
            #print(yScale,xScale)
            for i in range(h):
                for j in range(w):
                    #print(i,j)
                    narray[i,j]=oarray[int(i/yScale),int(j/xScale)]
            #print("dtype",narray.dtype)
            #nimage=Image.fromarray(narray)
            #Image._show(nimage.astype(np.uint8))
            dict[oimg]=narray
        return dict



    def crop(self, id1, id2, id3, id4):
        dict={}
        for oimg in self.flist:
            value=[0,0,0]
            oarray=Image.open(oimg)
            oarray=np.array(oarray)

            '''x=id4[0]-id1[0]+1
            y=id2[1]-id1[1]+1
            narray=np.full((x,y),0)

            for i in range(id1[0],id4[0]+1):
                for j in range(id1[1],id2[1]+1):
                    narray[i-id1[0]][j-id1[1]]=oarray[i][j]
            dict[oimg]=narray
            nimg=Image.fromarray(narray)
            Image._show(nimg)
        return dict
'''
            #x = id4[1] - id1[1] + 1
            #y = id2[0] - id1[0] + 1

            x = id1[1] - id4[1] + 1
            y = id2[0] - id1[0] + 1

            if len(oarray.shape)>2:
                narray=np.zeros((x,y,3),np.uint32)
            else:
                narray = np.zeros((x, y),np.uint32)

            #for i in range(oarray.shape[0]-1-id1[1],oarray.shape[0]-1-id4[1] + 1):
            for i in range(id4[1],id1[1] + 1):
                for j in range(id4[0], id3[0] + 1):
                    if len(oarray.shape)>2:
                        for k in range(3):
                            #value[k]=oarray[i][j][k]
                            value[k] = oarray[oarray.shape[0]-1-i][j][k]
                        #narray[i +id4[1]-oarray.shape[0]][j - id4[0]]=[value[0],value[1],value[2]]
                        narray[x-1-i + id4[1]][j - id4[0]] = [value[0], value[1], value[2]]
                    else:
                        narray[x-1-i+id4[1]][j - id4[0]] = oarray[oarray.shape[0]-1-i][j]
            dict[oimg] = narray
            #nimg = Image.fromarray(narray.astype(np.uint8))
            #Image._show(nimg)
        return dict

    def crop1(self,i1,i2,i3,i4):
        result=[]
        h=i4[0]-i2[0]+1
        w=i2[1]-i1[1]+1
        for img in self.flist:
            oimg=Image.open(img)
            oarray=np.array(oimg)
            narray=np.zeros((h,w))

            for i in range(h):#i1[0],i3[0]+1
                for j in range(w):#i1[1],i2[1]+1
                    narray[i][j]=oarray[i1[0]+i][i1[1]+j]
            result.append(narray)
        return result

    def translate(self,tx,ty):
        result=[]
        for img in self.flist:

            oimg=Image.open(img)
            oarray=np.array(oimg)

            h=oarray.shape[0]+ty
            w=oarray.shape[1]+tx

            narray=np.zeros(shape=(h,w))

            for i in range(np.shape(oarray)[0]):
                for j in range(np.shape(oarray)[1]):

                    narray[i][tx+j]=oarray[i][j]

            result.append(narray)
        return result


    def blur(self):
       dict={}
       for oimg in self.flist:
        a = Image.open(oimg)

        a = np.array(a, dtype='int64')
        #print("a:",a)
        b = a.copy()
        #print("b:", b)
        # print(b.shape)
        if len(b.shape)>2:
            img = np.zeros(((b.shape[0], b.shape[1],3)),np.uint32)
        else:
            img = np.zeros(((b.shape[0],b.shape[1])), dtype='int32')
        #img = np.zeros(((b.shape[0] - 2, b.shape[1] - 2)), dtype='int32')    without padding
        # print("imgshape",img.shape)

        if len(a.shape)>2:
            zero = np.zeros((1, b.shape[1],3),np.uint32)
            #print("bshape1",b.shape)
            b = np.append(zero, b, axis=0)
            b = np.append(b, zero, axis=0)
            zero = np.zeros((b.shape[0],1, 3), np.uint32)
            b = np.append(zero, b, axis=1)
            b = np.append(b, zero, axis=1)
            #b = b.T
            #print("bshape2",b.shape)
            #print(b[:][-1])
            # padding
            '''zero = np.zeros((1, b.shape[1],3),np.uint8)
            b = np.append(zero, b, axis=0)
            b = np.append(b, zero, axis=0)
            b = b.T'''
        else:
            #padding
            zero=np.zeros((1,b.shape[1]))
            b=np.append(zero,b,axis=0)
            b=np.append(b,zero,axis=0)
            b=b.T
            #padding
            zero = np.zeros((1, b.shape[1]))
            b = np.append(zero, b, axis=0)
            b = np.append(b, zero, axis=0)
            b=b.T
        #print("b",b[1][1],b[1][2], b[1][0], b[0][1], b[0][2], b[0][0], b[2][1],b[2][0], b[2][2])
        value=[0,0,0]

        #print("hi")
        for i in range(1, b.shape[0] - 1):
                for j in range(1, b.shape[1] - 1):
                    if len(a.shape)>2:
                        for k in range(3):
                            value[k]=b[i][j][k] + b[i][j + 1][k] + b[i][j - 1][k] + b[i - 1][j][k] + b[i + 1][j][k]+ b[i - 1][j-1][k]+ b[i - 1][j+1][k]+ b[i + 1][j+1][k]+ b[i + 1][j-1][k]
                        #print(value)
                        img[i - 1][j - 1] =value
                        #print("img{}{}:".format(i-1,j-1),img[i-1][j-1])
                    else:
                        img[i - 1][j - 1] = b[i][j] + b[i][j + 1] + b[i][j - 1] + b[i - 1][j] + b[i + 1][j]+ b[i - 1][j-1]+ b[i - 1][j+1]+ b[i + 1][j+1]+ b[i + 1][j-1]
                # print(i,j,b[i][j]+b[i][j+1]+b[i][j-1]+b[i-1][j]+b[i+1][j])
        #print("hi "+oimg,img[0,0])
        img=img//9

        #print("hi " + oimg, img[0, 0])
        dict[oimg]=img
        #nimg=Image.fromarray(img.astype(np.uint8))
        #Image._show(nimg)
       return dict

    def edge_detection(self):
        dict={}
        for oimg in self.flist:
            oarray=Image.open(oimg)
            oarray=np.array(oarray)
            #print(oarray)
            #padding
            b=oarray.copy()
            if len(oarray.shape)>2:
                zero = np.zeros((1, b.shape[1],3), dtype='int32')
                b = np.append(zero, b, axis=0)
                b = np.append(b, zero, axis=0)
                zero=np.zeros((b.shape[0],1,3),dtype='int32')
                b=np.append(zero,b,axis=1)
                b=np.append(b,zero,axis=1)
            else:
                zero=np.zeros((1,b.shape[1]),dtype='int32')
                b=np.append(zero,b,axis=0)
                b=np.append(b,zero,axis=0)
                b=b.T
                #padding
                zero = np.zeros((1, b.shape[1]),dtype='int32')
                b = np.append(zero, b, axis=0)
                b = np.append(b, zero, axis=0)
                b=b.T
            #print(oarray.shape[0])
            if len(oarray.shape)==2:
                narrayx=np.zeros((oarray.shape[0],oarray.shape[1]),np.uint32)
                narrayy=np.zeros((oarray.shape[0],oarray.shape[1]),np.uint32)
            else:
                narrayx = np.zeros((oarray.shape[0], oarray.shape[1],3), np.uint32)
                narrayy = np.zeros((oarray.shape[0], oarray.shape[1],3), np.uint32)
            #Gx=np.array([-1,0,1],[-2,0,2],[-1,0,1])
            #Gy=np.array([1,2,1],[0,0,0],[-1,-2,-1])

            value=[0,0,0]
            for i in range(1,b.shape[0]-1):
                for j in range(1,b.shape[1]-1):
                    if len(oarray.shape)>2:
                        for k in range(3):
                            value[k]=b[i-1][j-1][k]*(-1)+b[i-1][j+1][k]+b[i][j-1][k]*(-2)+b[i][j+1][k]*(2)+b[i+1][j-1][k]*(-1)+b[i+1][j+1][k]
                        narrayx[i-1][j-1]=value
                    else:
                        narrayx[i-1][j-1]=b[i-1][j-1]*(-1)+b[i-1][j+1]+b[i][j-1]*(-2)+b[i][j+1]*(2)+b[i+1][j-1]*(-1)+b[i+1][j+1]
            value=[0,0,0]
            for i in range(1,b.shape[0]-1):
                for j in range(1,b.shape[1]-1):
                    if len(oarray.shape)>2:
                        for k in range(3):
                            value[k]=b[i-1][j-1][k]*(1)+b[i-1][j][k]*2+b[i-1][j+1][k]+b[i+1][j-1][k]*(-1)+b[i+1][j][k]*(-2)+b[i+1][j+1][k]*(-1)
                        narrayy[i-1][j-1]=value
                    else:
                        narrayy[i-1][j-1]=b[i-1][j-1]*(1)+b[i-1][j]*2+b[i-1][j+1]+b[i+1][j-1]*(-1)+b[i+1][j]*(-2)+b[i+1][j+1]*(-1)

            gradient=np.sqrt(np.square(narrayx)+np.square(narrayy))
            #gradient.dtype='int64'
            #print("Before",gradient)
            #nimg = Image.fromarray(gradient.astype(np.uint8))
            #Image._show(nimg)
            #gradient*=255.0/gradient.max()
            #print("After",gradient)
            #nimg=Image.fromarray(gradient.astype(np.uint8))
            #Image._show(nimg)
            dict[oimg]=gradient
        return dict

    def __getitem__(self):
        dict={}
        for oimg in self.flist:
            dict[oimg]=Image.open(oimg)
        return dict

    def rgb2gray(self):
        dict={}
        for oimg in self.flist:
            oarray=Image.open(oimg)
            oarray=np.array(oarray)
            if len(oarray.shape)==2:
                dict[oimg]=oarray

            else:
                garray=np.full(oarray.shape,0.0)

                garray=oarray[:,:,0]*(0.2989)+oarray[:,:,1]*(0.5870)+oarray[:,:,2]*(0.1140)
                dict[oimg]=garray
                #nimg=Image.fromarray(garray)
                #Image._show(nimg)
        return dict

    def rotate(self,theta):
        dict={}
        #print("hi")
        for oimg in self.flist:
            oarray=Image.open(oimg)
            oarray=np.array(oarray)
            nh=round(abs(oarray.shape[0]*np.cos(theta)+abs(oarray.shape[1]*np.sin(theta))))+1
            nw=round(abs(oarray.shape[1]*np.cos(theta))+abs(oarray.shape[0]*np.sin(theta)))+1

            if len(oarray.shape)>2:
                narray=np.zeros((nh,nw,3),np.uint32)
            else:
                narray=np.full((nh,nw),0)

            ch1=round((oarray.shape[0]+1)/2)-1
            cw1=round((oarray.shape[1]+1)/2)-1

            ch2=round((narray.shape[0]+1)/2)-1
            cw2=round((narray.shape[1]+1)/2)-1

            for i in range(oarray.shape[0]):
                for j in range(oarray.shape[1]):
                    height=oarray.shape[0]-1-i-ch1
                    width=oarray.shape[1]-1-j-cw1

                    new_height=round(height*np.cos(theta)-width*np.sin(theta))
                    new_width=round(width*np.cos(theta)+height*np.sin(theta))

                    new_width1=cw2-new_width
                    new_height1=ch2-new_height
                    if len(oarray.shape)>3:
                        value=[0,0,0]
                        for k in range(3):
                            value[k]=oarray[i,j,k]
                        narray[new_height1,new_width1]=value
                    else:
                        narray[new_height1, new_width1] = oarray[i,j]

            dict[oimg]=narray
            #nimg=Image.fromarray(narray.astype(np.uint8))
            #Image._show(nimg)
        return dict


l='<ENTER PATH FOR IMAGES DIRECTORY>'
obj1=preprocess(l)  #instantiate object of class
#now perform operations using obj1


