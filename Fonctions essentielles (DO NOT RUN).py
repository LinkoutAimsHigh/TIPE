#  imports globaux

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random as rd
import scipy
import cv2
import tensorflow as tf 

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from numpy.random import RandomState

from IPython.display import clear_output

# Etude première : Gouttes

# Traitement des données

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


# Préparation des données pour l'IA 

Base = "TxtDataOut"

Shapes = []
DataFrame,LabelFrame = [],[]
for file in os.listdir(Base): # Ouverture des données
    df = np.load(Base+ '/'+file)
    Shapes.append(df.shape[1])
    DataFrame.append(df)
    LabelFrame.append([round(float(c.replace(',','.'))/100,3) for c in file[:-4].split()])


# travail des données pour enlever les données peu pertinentes
DelNumb = 3
Data,Label = DataFrame.copy(),LabelFrame.copy()

Deletor = sorted([(i,Shapes[i]) for i in range(len(Shapes))],key=lambda x : x[1])[:DelNumb]
for j,k in sorted(Deletor,reverse=True):
    Label.pop(j)
    Data.pop(j)

LABELS = np.array(Label)
DATA = []

LSNb = min([d.shape[1] for d in Data])

for df in Data:
    Ndf = np.ndarray((3,LSNb))
    
    for j,d in enumerate(df) :
        Nbo = d.tolist()
        Indexes = rd.sample(range(len(Nbo)),LSNb)
        Nbn = [Nbo[i] for i in range(len(Nbo)) if i in Indexes]
        Ndf[j] = Nbn
    DATA.append(Ndf)
DATA = np.array(DATA)


# Utilisation de PCA pour améliorer les performances
Datatrain = [{"name":"Number" },{"name":"Speed"},{"name":"Excentricity"}]

PCASize = 15
TTS = rd.sample(range(len(DATA)),int(len(DATA)*(1-.2)))
for i in range(3):
    d = DATA[:,i,:]
    pca = PCA(PCASize)
    pca.fit(d)
    dt = pca.transform(d)

    Datatrain[i]["X_train"],Datatrain[i]["X_test"] = np.array([dt[i] for i in range(len(dt)) if i in TTS]), np.array([dt[i] for i in range(len(dt)) if i not in TTS]) #5464021513.2

# -----------------------------------------------------------------------

#Etude Deuxième : Titrage

# Simulation théorique des courbes de Titrage

Ke = 10**(-14)

def neutralitePolyAcide(pH,C,V0,Ct,Vt,pKa): 
    h = 10**(-pH)
    Ka = [10**(-pka) for pka in pKa]
    neutre = (h - Ke/h)*(V0 + Vt) + Ct*Vt
    for i in range(len(C)) :
        neutre = neutre - C[i]*V0/(1+h/Ka[i])
    return neutre

def calcul_pHPolyAcide(C,V0,Ct,Vt,pKa): 
    a = 0
    b = 14
    while b-a > 0.01 : 
        m = (a + b)/2
        if neutralitePolyAcide(a,C,V0,Ct,Vt,pKa)*neutralitePolyAcide(m,C,V0,Ct,Vt,pKa) <= 0 :
            b = m
        else :
            a = m
    return m

def titragePolyAcide(pKa, C, V0, Ct, Vmax, pas) :
    VSim = [] # tableau de valeur pour stocker les volumes
    PHSim = [] # tableau de valeur pour stocker les pH correspondants
    v = 0 # volume "courant"
    while v <= Vmax : 
        VSim.append(v)
        c = 10**(-calcul_pHPolyAcide(C, V0, Ct, v, pKa))
        PHSim.append(-np.log10(c*V0/(V0+v)))
        v = v + pas
    return VSim, PHSim

Concentrations = np.array([[float(a) for a in line.split()] for line in open('../Données expérimentales/Labels.txt')][4:])/100

pKas = [2.87,4.2,4.76, 5.48 ]
V0,Vmax,Ct,DV = 10,10,0.01,0.25

DATA = []

for cc in Concentrations : 
    X, Y = titragePolyAcide(pKas, cc, V0, Ct, Vmax, DV)
    DATA.append(Y)
    plt.plot(X[:-1],Y[:-1])
plt.show()

DATA = np.array(DATA)
LABELS = np.array(Concentrations)*100

pca = PCA(5)
pca.fit(DATA)
DataTrain = pca.transform(DATA)

#X_train,X_test,Y_train, Y_test = train_test_split(DataTrain/14, Labels, test_size=0.2)

X_train,X_test,Y_train, Y_test = train_test_split(DataTrain/max([max(DataTrain[:,i]) for i in range(DataTrain.shape[1])]), LABELS, test_size=0.2)


#Visualisation des courbes de titrage


base = '../Données expérimentales/T'
Data = []


for i in range(4,len(os.listdir('../Données expérimentales/'))-1):
    #if i in Privation:
    #    continue
    File = open(base+str(i)+'.txt')
    data = np.array([ f.split(':') for f in File])

    V,pH = np.asarray(data[:,3],dtype=float), np.asarray(data[:,5],dtype=float)

    plt.ylim(2,12.5)
    plt.plot(V,pH)


plt.show()


#------------------------------------------------

#Réseau neuronal utilisé dans les différents cas


#Fonctions utiles

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return 1 / (np.exp(x)+np.exp(-x)+1)

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1,0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

# Xavier initialization function to not have weight too small that can cause instability
def xavier_init(size_in, size_out, random_state = None):
    rng = random_state
    limit = np.sqrt(6 / (size_in + size_out))
    return rng.uniform(-limit, limit, (size_in, size_out))

def Loading(tx):
    # clear_output(wait=True)
    Loadingbar = ''.join(["=" if (i < int(tx*100)) else " " for i in range(100)])
    print("[" + "\033[32m" +  Loadingbar +'\033[0m' + "]")

def Accuracy(Nw,X,Y):
    predictions = Nw.forward(X)
    return round(np.mean((Y -predictions)**2 <= (0.05)**2),2) + 0.01*round(np.mean((Y -predictions)**2 <= (0.02)**2),2) #, np.mean([round(Y[i],1)==round(predictions[i],1) for i in range(len(Y))])


class Activation_Function:
    
    def __init__(self,func,func_derivative, name = None):
        self.func = func
        self.derivative = func_derivative
        self.name = name 
        


class Layer:
    def __init__(self,in_size,out_size, random_state = None):
        rng = random_state if random_state else np.random #setting the random seed instance for stability and reproductibility
        
        self.weights = xavier_init(in_size, out_size, random_state = rng)#np.random.randn(in_size,out_size) #xavier_init(in_size, out_size)
        self.bias = np.zeros((1,out_size))
        #self.delta = np.zeros((19,1))

    def Calculate_error(self, out_weights,out_delta,activation_derivative):

        self.delta =  out_delta.dot(out_weights.T) * activation_derivative(self.output)
        return self.delta

    def Calculate_output_error(self,y,output,activation_derivative):
        ### Only use if the layer is the output layer (might do a separate func one day)
        self.delta = (y-output)*activation_derivative(output)
        return self.delta

    def update(self,error, inputs, learning_rate):
        self.weights += inputs.T.dot(error) * learning_rate
        self.bias += np.sum(self.delta, axis=0, keepdims=True) * learning_rate

    def forward(self, activation, X):
        self.inputs = np.dot(X,self.weights) + self.bias
        self.output = activation(self.inputs)
        return self.output


class Network:
    def __init__(self,input_size,hidden_size,output_size,Name = "Network", random_state = None,activation = Activation_Function(relu,relu_derivative)):
        self.name = Name
        self.random_state = random_state
        
        # Initializing layers
        self.hidden_layer = Layer(input_size,hidden_size, random_state = random_state)
        self.output_layer = Layer(hidden_size,output_size, random_state = random_state)      

        self.out_delta = self.hidden_delta = 0
    
        self.activation = activation
    

    def forward(self,X):
        self.hidden_layer.forward(self.activation.func,X)

        self.output_layer.forward(relu,self.hidden_layer.output)

        self.output = self.output_layer.output

        return self.output_layer.output
    
    def loss(self,y,output):
        return (y-output)#/(abs(y-output))#**0.5

    

    def learn(self,X,y,learning_rate,momentum):

        #Calculating error
        self.out_delta = self.output_layer.Calculate_output_error(y,self.output_layer.output,relu_derivative) + self.out_delta*momentum
        self.hidden_delta = self.hidden_layer.Calculate_error(self.output_layer.weights,self.output_layer.delta,self.activation.derivative) + self.hidden_delta*momentum

        # Update weights and biases

        self.output_layer.update(self.out_delta,self.hidden_layer.output,learning_rate)
        self.hidden_layer.update(self.hidden_delta,X,learning_rate)

    def train(self,X,y,epochs,learning_rate,momentum,train_steps = 100, Training_Info= True, random_state = None):
        History = []
        rng = random_state
        for epoch in range(epochs):
            
            self.forward(X)
            self.learn(X,y,learning_rate,momentum)

            loss = np.mean(np.square(y - self.output))
            History.append(loss)   

            if Training_Info and epoch % train_steps == 0 :
                
                clear_output(wait=True)
                print(f"Training {self.name} \n Epoch  : {epoch} / {epochs} \n Loss : {loss} \n Advancement : {epoch/epochs*100} %")
                Loading(epoch/epochs)

        if Training_Info:
            loss = np.mean(np.square(y - self.output))
            clear_output(wait=True)
            print(f"Training {self.name} \n Epoch  : {epochs} / {epochs} \n Loss : {loss} \n Advancement : {100} %")
            Loading(1)
        return History
        



#Exemple d'entrainement :

# Initialisation:

Acides = ['acide_chloroacétique', 'acide_benzoique', 'acide_éthanoique', '4_aminophénol']

Models_val = [0 for i in Acides]

Histories = [None for i in Acides]

Relu,Tan,Sig = Activation_Function(relu,relu_derivative,name='relu'),Activation_Function(tanh,tanh_derivative,name = 'tanh'),Activation_Function(sigmoid,sigmoid_derivative,name = 'sigmoid')


from itertools import product
#np.random.seed(42)
# Hyperparameter ranges
i_size,o_size = 5,1 
epoch_options =[8000]#[8000, 2000, 3000, 5000, 10000]# [8000, 10000,9000,7000]#, 30000], 35000, 40000]  # Possible values for epochs
learning_rate_options = [0.0001, 0.0005, 0.001, 0.005,0.00005,0.01]# 0.01, 0.05, 0.09, 0.1]  # Possible values for learning rates
momentum_options = [0.5,0.2, 0.4, 0.6, 0.8]  # Possible values for momentum
h_size_options = [3,4,5,6,7] #[6]
activation_options = [Tan,Sig,Relu]

N_iter =  len(h_size_options)*len(epoch_options)*len(learning_rate_options)*len(momentum_options)*len(activation_options)

# Store best parameters and scores for each acid
best_params = {}
best_scores = {}
Models = [0 for i in range(len(Acides))]
# Perform grid search for each acid
for i, acid in enumerate(Acides):
    print(f"\nTuning hyperparameters for acid: {acid}")
    
    best_params[acid] = None
    best_scores[acid] = -float('inf')#-float('inf')  # Use 'inf' because we are minimizing the loss
    #first = True
    # Grid search over all hyperparameter combinations
    iteration = 0
    for h_size, epochs, learning_rate, momentum,activation in product(h_size_options, epoch_options, learning_rate_options, momentum_options,activation_options):
        clear_output(wait=True)
        Loading(iteration/N_iter)
        print(f"Testing combination for {acid}: Epochs={epochs}, Learning Rate={learning_rate}, Momentum={momentum}, Activation={activation.name}")
        iteration+=1
        rng = RandomState(42)
        # Initialize and train the model for the current acid
        model = Network(input_size=i_size, output_size=o_size, hidden_size=h_size, Name=acid, random_state = rng,activation = activation)
        history = model.train(
            X_train,
            np.array([[Y_train[j][i]] for j in range(len(Y_train))]),
            epochs=epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            Training_Info = False,
            train_steps=1000, 
            random_state = rng
        )
        """
        val_predictions = model.forward(X_test)
        val_loss = np.mean((np.array([[Y_test[j][i]] for j in range(len(Y_test))]) - val_predictions.T) ** 2)
        
        # Keep track of the best hyperparameters for this acid
        if val_loss < best_scores[acid]:
            best_scores[acid] = val_loss
            best_params[acid] = (epochs, learning_rate, momentum)
        """
        # Evaluate the model on the validation set
        current_accuracy = Accuracy(model,X_test,np.array([[Y_test[j][i]] for j in range(len(Y_test))]))
        
        
            # Keep track of the best hyperparameters for this acid
        if current_accuracy > best_scores[acid]:
            best_scores[acid] = current_accuracy
            best_params[acid] = (epochs, learning_rate, momentum, h_size,activation)
            Models[i] = model
            first = False
    print(f"Best Hyperparameters for {acid}: Epochs={best_params[acid][0]}, Learning Rate={best_params[acid][1]}, Momentum={best_params[acid][2]}, function={best_params[acid][4].name}")
    print(f"Best Validation Loss for {acid}: {best_scores[acid]:.4f}")
    
# Final results for all acids
print("\nFinal Best Hyperparameters for Each Acid:")
for acid in Acides:
    print(f"{acid}: Epochs={best_params[acid][0]}, Learning Rate={best_params[acid][1]}, Momentum={best_params[acid][2]}, Validation accuracy={best_scores[acid]:.4f}, h_size={best_params[acid][3]}, function={best_params[acid][4].name}")


#Vérification et Visualisation des résultats:

predictions, = np.array([Nw.forward(X_test) for Nw in Models]).T  

i=0
for pred in predictions:
    print(str([float(round(pd,2)) for pd in pred]) + "vs" + str(Y_test[i])) 
    i +=1

for i in range(len(predictions.T)):
    plt.scatter(Y_test.T[i],predictions.T[i], marker='x',label = Acides[i])

x = np.linspace(0,1,100)

plt.plot(x,x, color = 'red')
plt.plot(x[2:],(x-0.05)[2:], '--b')
plt.plot(x,x+0.05, '--b')

plt.xlim(0,0.85)
plt.ylim(0,0.85)

plt.ylabel('Prédictions')
plt.xlabel('Concentrations réelles')

plt.legend()
plt.show()



#-----------------------------------------------------

#Troisième Etude : Identification des instruments

#fonctions utiles

def Linearize_audio(aud_data):
    if len(aud_data.shape)>1:
        linear_audio = (aud_data[:,0]+aud_data[:,1])/2
    else:
        linear_audio = aud_data
    
    linear_audio = np.array(linear_audio)

    msk = (abs(linear_audio[:])>max(linear_audio)*0.1)

    return linear_audio[msk]

def Spectrum_Treatment(data,cut):
    N = len(data)
    Y = np.fft.fft(data)
    Y = abs(Y[:N//cut])
    frequency = rate*np.arange((N//cut))/N
    return Y

def Accuracy(Nw,X,Y):
    predictions = Nw.forward(X)
    return round(np.mean((Y -predictions)**2 <= (0.05)**2),2) + 0.01*round(np.mean((Y -predictions)**2 <= (0.02)**2),2) #, np.mean([round(Y[i],1)==round(predictions[i],1) for i in range(len(Y))])

def minuscules(String):
    m1,m2 =ord('A'),ord('a')
    dm = m2-m1
    FS = ''
    for s in String:
        if ord(s)<m2:
            FS = FS+chr(ord(s)+dm)
        else:
            FS = FS +s
    return FS

def Reverse(label):
    for i in range(len(classes)):
        if classes[i]==label:
            return i

#Traitement des données

#Dimension parameters
NbSegments = 50
FreqHeight = 40
Super = 0.5
DATA = []
LABELS = []

i = 0
cut = 8  # paramètre d'ajustement du spectrogramme

for file in os.listdir("Audio_Trainer"):
    
    rate,aud_data = scipy.io.wavfile.read("Audio_Trainer/"+file)
    aud_data = Linearize_audio(aud_data)

    N = len(aud_data)

    Seglen =N//NbSegments
    Super_Seglen = round(Super*N)//NbSegments

    Segments = [aud_data[i*Seglen:(i+1)*(Seglen)+Super_Seglen]  for i in range(NbSegments-1)] #if i<NbSegments-1 else aud_data[i*Seglen:(i+1)*(Seglen)] 
    Segments = np.array(Segments)

    Ys = np.array([Spectrum_Treatment(data,cut) for data in Segments])

    Max = max([max(y) for y in Ys])
    Ys/=Max

    if Ys.shape[1]<FreqHeight :
        i+=1
        print('passed one more : ',i)
        continue

    Seglen = len(Ys[0])//FreqHeight
    Lines = np.array([[np.max(Y[i*Seglen:(i+1)*Seglen]) for i in range(FreqHeight)] for Y in Ys]).T

    DATA.append(Lines.flatten())
    LABELS.append(minuscules(file.split()[0]))


DATA = np.array(DATA)


classes = []
for l in LABELS:
    if l not in classes:
        classes.append(l)

Numeric_classes = [i for i,j in enumerate(classes)]

Labels = np.array([Reverse(l) for l in LABELS])

X_train,X_test,Y_train, Y_test = train_test_split(DATA, Labels, test_size=0.2) #/max([max(DATA[:,i]) for i in range(DATA.shape[1])])



#Entrainement avec tensorflow

import tensorflow as tf

# un exemple de réseau neuronal

inputs = tf.keras.Input(shape=(DATA.shape[1],))
x = tf.keras.layers.Dense(256,activation='tanh')(inputs)
x = tf.keras.layers.Dense(128,activation='tanh')(x)
x = tf.keras.layers.Dense(64,activation='tanh')(x)
outputs=tf.keras.layers.Dense(len(classes),activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.compile(loss = "sparse_categorical_crossentropy",optimizer="SGD")

history = model.fit(X_train,Y_train,epochs=2000,verbose = 0)

#Vérifications

predictions = model.predict(X_test)

for i in range(len(X_test)):
    print(classes[np.argmax(predictions[i])],classes[Y_test[i]])

print(np.mean([classes[np.argmax(predictions[i])]==classes[Y_test[i]] for i in range(len(X_test))]))




