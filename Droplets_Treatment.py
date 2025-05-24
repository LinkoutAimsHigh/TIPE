import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rd 


def optimized_treatment(image, cc=70):
    # Calculate the threshold mask using NumPy array operations
    red_mask = (image[:,:,2]-cc > image[:,:,0] ) & (image[:,:,2]-cc > image[:,:,1] ) & (image[:,:,2]>cc)
    # Set the pixel values based on the threshold mask
    image[red_mask] = [255, 255, 255]
    image[~red_mask] = [0, 0, 0]
    return image
def mini(L):
    a = L[0]
    x = 0
    for i in range(len(L)):
        if L[i]<a:
            a=L[i]
            x = i
    return x
def Nearest(pts1,pts2):
    if len(pts1)==0 or len(pts2)==0:
        return []
    Near = []
    for pt in pts1:
        dist =[]
        for pt2 in pts2:
            dist.append(np.sqrt((pt[0]-pt2[0])**2+(pt[1]-pt2[1])**2))
        Near.append([pt,pts2[mini(dist)]])
    return Near
def dist(pt1,pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 +(pt1[1]-pt2[1])**2)
def Verify(Circles):
    Tocheck = Circles.copy()
    while Tocheck!=[]:
        c = Tocheck.pop()
        i=0
        while i<len(Circles):
            if dist(c,Circles[i])<c[2] and c!=Circles[i]:
                Circles.pop(i)
                i-=1
            i+=1
    return Circles
def DontCollide(Circles,pt):
    for c in Circles:
        if dist(c,pt)<[c[2]+pt[2]]:
            if c[2]>pt[2]:
                return False
            else :
                del c
    return True
def Contours(frame,center):
    contours, hierarchy = cv2.findContours( cv2.Canny(optimized_treatment(frame),10,255),  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    Circles = []
    for cnt in contours:
        coordx,coordy = [cnt[i][0][0] for i in range(len(cnt))],[cnt[i][0][1] for i in range(len(cnt))]
        x,y = np.mean(coordx),np.mean(coordy)
        r = max([np.sqrt((coordx[i]-x)**2+(coordy[i]-y)**2) for i in range(len(cnt))])
        bd = (round(x),round(y),round(r),round(dist((x,y),center)))
        if r>2 and DontCollide(Circles,bd):
            Circles.append(bd)
    To_Delete = []
    for i in range(len(Circles)):
        for j in range(i+1,len(Circles)):
            if dist(Circles[i],Circles[j])<[Circles[i][2]+Circles[j][2]]:
                if Circles[i][2]<Circles[j][2]:
                    To_Delete.append(Circles[i][2])
                else : 
                    To_Delete.append(Circles[j][2])
    Better_Circles = [c for c in Circles if (c not in To_Delete)]
    return Better_Circles,len(Better_Circles)
def blur(x,y=5):
    blurry = np.array([np.exp(-abs(i/y)) for i in range(-x,x+1)])
    blurry = blurry/sum(blurry)
    return blurry

Text = True
Figures = False
In,TxtOut,FigOut = "Verif data","Verif data","FigDataOut2"

for file in os.listdir(In) :
    rpath,npathtxt,npathfig = In + "/"+file, TxtOut + "/"+file[:-4],FigOut + "/"+file[:-3]+"png"
    ###if file[:-3]+"txt" in os.listdir(TxtOut):
        #print("Tâche déjà effectuée")
        #continue
    Name = rpath
    video  = cv2.VideoCapture(Name)
    video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)
    #for i in range(7*int(video_length)//8):
    #    ok,frame = video.read()
    width,height = video.get(cv2.CAP_PROP_FRAME_WIDTH),video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center = (width/2,height/2)
    fps = video.get(cv2.CAP_PROP_FPS)
    video_length = video.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    ok,frame = video.read()
    if not ok:
        print("Launch fail")
    Circ1,nbdrops = Contours(frame,center)
    Nbdroplets,Speed,Excentricity = [nbdrops],[],[np.mean([C[3] for C in Circ1]) if len(Circ1)!=0 else 0]
    i=1
    bboxs = []
    while True:
        ok,frame = video.read()
        if not ok:
            print("Video ended")
            break
        Circ2,drops = Contours(frame,center)

        Excentricity.append(np.mean([C[3] for C in Circ2]) if len(Circ2)!=0 else 0)
        Nbdroplets.append(drops)
        Nearpoints = Nearest(Circ1,Circ2)
        Spd = []
        for n in Nearpoints:
            Spd.append(dist(n[0],n[1])/fps)
        Speed.append(np.mean(Spd) if len(Spd)!=0 else 0)
        Circ1=Circ2
        i+=1
    video.release()

    Nblur1,Nblur2 = 20,20
    T = [(j+1)/fps for j in range(len(Nbdroplets))]
    NbDropletsPropre = [int(pt) for pt in np.convolve(Nbdroplets,blur(Nblur1))]
    SpeedPropre =  [pt for pt in np.convolve(Speed,blur(Nblur2,15))]
    # print(len(NbDropletsPropre),len(SpeedPropre),len(Excentricity))
    Points = np.array([NbDropletsPropre[(Nblur1+15):-(Nblur1+15)-1],SpeedPropre[(Nblur2+15):-(Nblur2+15)],Excentricity[15:-16]])
    if Text :
        np.save(npathtxt,Points)
    if Figures : 
        fig,ax = plt.subplots(3,1)
        ax[0].plot(T[15:-15],NbDropletsPropre[(Nblur1+15):-(Nblur1+15)])
        ax[0].set_ylabel('Number of droplets')
        ax[1].plot(T[15:-16],SpeedPropre[(Nblur2+15):-(Nblur2+15)])
        ax[2].set_xlabel('Time(s)')
        ax[1].set_ylabel('Speed(pixels per second)')
        ax[2].plot(T[15:-16],Excentricity[15:-16])
        ax[2].set_ylabel('Excentricity')
        plt.suptitle(Name)
        plt.savefig(npathfig)
        plt.close()
