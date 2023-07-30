
import matplotlib.pyplot as plt 
import random as rnd
import numpy as np
import cv2 as cv
import os
import math



class processimage():

    def __init__(self):
        self.x=0
        self.y=0 
        self.w=0
        self.h=0
    
    def background(self,thr):

        """Borramos en el fondo de la foto"""

        hist1 = cv.calcHist([self.src_copy2], [2], None, [256], [0, 256])
        brillo=np.argmax(hist1)

        if brillo>100:
            self.imgResult[np.all(self.imgResult>=thr, axis=2)] = 0
            self.imgT.append(self.imgResult.copy())
        else:
            self.imgResult[np.all(self.imgResult<=thr, axis=2)] = 0
            self.imgT.append(self.imgResult.copy())

    def filter(self):

        """Aplicamos un filtro laplaciono"""
        
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        imgLaplacian = cv.filter2D(self.src, cv.CV_32F, kernel)
        sharp = np.float32(self.src)
        self.imgResult = sharp - imgLaplacian
        #Convertimos devuelta a escala de grises
        self.imgResult = np.clip(self.imgResult, 0, 255)
        self.imgResult = self.imgResult.astype('uint8')
        self.imgT.append(self.imgResult.copy())
        imgLaplacian = np.clip(imgLaplacian, 0, 255)
        imgLaplacian = np.uint8(imgLaplacian)

    def process(self,name,foto=True):

        """ Realiza el procesamiento de las imagenes y su vecotrización """

        self.imgT=[]
        if foto:
            self.src = cv.imread(name)
            self.src_copy = cv.imread(name)
            self.src_copy2 = cv.imread(name)  
        else:
            self.src=name
            self.src_copy = name
            self.src_copy2 =name

        self.imgT.append(self.src.copy())
        src_copy2=cv.cvtColor(self.src_copy2,cv.COLOR_BGR2HSV)
        hist = cv.calcHist([src_copy2], [1], None, [256], [0, 256])
        saturacion=np.argmax(hist)
        hist = cv.calcHist([src_copy2], [2], None, [256], [0, 256])
        brillo=np.argmax(hist)
        thr=130
        if saturacion<=5 and brillo<=230:

            thr=120

        if saturacion<=5 and brillo>=230:

            thr=140

        if 5>saturacion and saturacion<=15:

            thr=80

        if saturacion>15 and brillo<200:

            thr=50
        if saturacion>15 and brillo>=200:

            thr=90

        if saturacion>230 and brillo<=200:

            thr=40

        self.filter()   
        self.background(thr)
        

        #Convertimos a escala de grises
        bw = cv.cvtColor(self.imgResult, cv.COLOR_BGR2GRAY)
        self.imgT.append(bw.copy())
        #Aplicamos el threshold
        _, bw = cv.threshold(bw, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        self.imgT.append(bw.copy())
        #Aplicando el distance Transform, podemos obtener los picos de la imagen
        # por lo que a partir de ahí podemos realizar una segmentación de la imagen basada en marcadores utilizando el watershed algorithm.
        dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
        # Normalizamos la imagen en el rango = {0.0, 1.0}
        # asi podemos visualizarla y aplicarle el threshold.
        cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
        self.imgT.append(dist.copy())
        _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
        # Dilatamos un poco la imagen transformada
        kernel1 = np.ones((3,3), dtype=np.uint8)
        dist = cv.dilate(dist, kernel1)
        dist_8u = dist.astype('uint8')
        # Buscamos los marcadores
        contours,_ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Creamos una imgane marcada para el watershed algoritmo
        markers = np.zeros(dist.shape, dtype=np.int32)
        # Rellenamos los marcadores

        for i in range(len(contours)):
            cv.drawContours(markers, contours, i, (i+1), -1)
        # Dibujamos el fondo de los marcadores
        cv.circle(markers, (5,5), 3, (255,255,255), -1)
        #Aplicamos el watersheld algorithm
        self.imgResult[bw==0] = 0
        cv.watershed(self.imgResult, markers)
        self.mark = markers.astype('uint8')
        self.mark = cv.bitwise_not(self.mark)
        kernel = np.ones((5,5), dtype=np.uint8)
        #Filtramos el ruido.
        self.mark=cv.morphologyEx(self.mark,cv.MORPH_CLOSE,kernel,iterations = 3)
        self.mark=cv.morphologyEx(self.mark,cv.MORPH_OPEN,kernel,iterations = 3)
        self.imgT.append(self.mark.copy())
        contour1,_ = cv.findContours(self.mark,1,2)
        self.cnt=[]

        if len(contour1)>0:
            a=0
            for i in range(len(contour1)):
                if len(contour1[i])>=len(contour1[a]):
                    a=i
            self.cnt=contour1[a]   
            self.cnt= cv.convexHull(self.cnt)
            cv.drawContours(self.src_copy,contour1, -1, (0,255,0), 5)
            self.imgT.append(self.src_copy.copy())
            self.x,self.y,self.w,self.h = cv.boundingRect(self.cnt)
           
    def data(self):


        s=cv.bitwise_and(self.src_copy,self.src_copy,mask = self.mark)
        s=cv.cvtColor(s,cv.COLOR_BGR2HSV)
        hist = cv.calcHist([s], [0], None, [256], [1, 256])
        color=np.argmax(hist)
        if color > 50:
            color=1
        if len(self.cnt)>0:
            area = cv.contourArea(self.cnt)
            perimetro = cv.arcLength(self.cnt,True)
            (x,y),radius = cv.minEnclosingCircle(self.cnt)
            rel_cir=math.pi*(radius**2)/area
            rel_per=math.pi*2*radius/perimetro
            rect = cv.minAreaRect(self.cnt)
            rel_ret=rect[1][0]*rect[1][1]/area
            M = cv.moments(self.cnt)
            return float(color), float(rel_cir),float(rel_per),float(rel_ret),float(M['nu20'])
        return 0,0,0,0,0

class data_plot():

    def __init__(self):
        pass

    def matplotlib_multi_pic1(self,s):

        """Función para plotear el tratamiento de las imagenes"""

        titles=['Img','L. filter','Sin BG','Gray','Threshold','Dist. T.','Watersheld','Contornos']
        aa=0
        for i in s:
            bb=0
            for j in i:
                title=titles[bb]
                plt.subplot(len(s),len(i),aa+1)
                j=cv.cvtColor(j,cv.COLOR_BGR2RGB)
                plt.imshow(j)
                if aa<len(i):
                    plt.title(title,fontsize=8)
                plt.xticks([])
                plt.yticks([])
                aa+=1
                bb+=1
        plt.show()

    def plt_basedatos(self,s):

        aa=0
        for i in s:
            if aa<30:
                plt.subplot(5,6,aa+1)
                i=cv.cvtColor(i[0],cv.COLOR_BGR2RGB)
                plt.imshow(i)
                plt.xticks([])
                plt.yticks([])
            aa+=1
        plt.show()

    def plotdata(self,data,cls):

        """"Plotea parámetros característicos de los datos"""

        color=['or','go','ko','yo','ob']
        for i in range(len(cls.x_T)):
            plt.plot(cls.x_T[i],cls.y_T[i],color[cls.cluster_T[i]])
        
        plt.xlabel("Color")
        plt.ylabel("Relación área círculo")
        plt.show()

        z_T=np.array(data['nu02'])
        z_T=z_T/max(z_T)
        for i in range(len(cls.x_T)):
            plt.plot(cls.x_T[i],z_T[i],color[cls.cluster_T[i]])
        
        plt.xlabel("Color")
        plt.ylabel("Momento nu20")
        plt.show()

        h_T=np.array(data['ret'])
        h_T=h_T/max(h_T)
        for i in range(len(cls.x_T)):
            plt.plot(cls.x_T[i],h_T[i],color[cls.cluster_T[i]])
        
        plt.xlabel("Color")
        plt.ylabel("Relación área rectángulo")
        plt.show()

        l_T=np.array(data['per'])
        l_T=l_T/max(l_T)
        for i in range(len(cls.x_T)):
            plt.plot(cls.x_T[i],l_T[i],color[cls.cluster_T[i]])
        
        plt.xlabel("Color")
        plt.ylabel("Relación perímetro")
        plt.show()

    def plotimage(self,tabla,n):
        
        """Ploteamos el procesamiento de las imagenens"""

        for i in tabla['imagen'].keys():
            for j in range(round(len(tabla['imagen'][i])/n)):
                self.matplotlib_multi_pic1(tabla['imagen'][i][j*n:(j+1)*n])

class clasificacion():

    def __init__(self,temple):

        self.temple=temple
        self.dd=data_plot()
        self.pr=processimage()

    def data_accond(self,data):

        """Acondicionamiento de los datos almacenados en Cache"""

        #x_T & y_T: Lista con la info de todas las frutas
        #cluster_T: Lista con la info de que fruta representa cada dato
        self.x_T=[]
        self.y_T=[]
        self.cluster_T=[]
        ii=0
        for j in data['data_x'].keys():
            self.x_T.extend(data['data_x'][j])
            self.cluster_T.extend([ii for i in range(len(data['data_x'][j]))])
            self.y_T.extend(data['data_y'][j])
            ii=ii+1

        self.x_T=np.array(self.x_T)
        self.y_T=np.array(self.y_T)
        self.x_Tmax=max(self.x_T)
        self.y_Tmax=max(self.y_T)

        #Normalizamos los datos
        self.x_T=self.x_T/self.x_Tmax
        self.y_T=self.y_T/self.y_Tmax

    def kmeans_init(self,plot=False,kn=4):

        """"Realiza la clusterización de la base de datos mediante el algoritmo Kmeans"""

        #Cluster inicial
        self.cluster=rnd.choices(range(kn),k=len(self.x_T))
        self.cluster,list_energy=self.temple.search(self.cluster,self.x_T,self.y_T,kn)
        #Aplicamos kmeans
        self.cluster,self.x_cluster,self.y_cluster=kmeans(self.x_T,self.y_T,self.cluster,kn=kn)

        if plot:
            #Ploteamos las imagenes vectorizadas
            print('BASE DE DATOS')
            self.color=['or','go','ko','yo','ob']
            for i in range(len(self.x_T)):
                plt.plot(self.x_T[i],self.y_T[i],self.color[self.cluster_T[i]])
            plt.xlabel('Color')
            plt.ylabel('Relación área círculo')
            plt.title('BASE DE DATOS')
            plt.show()
            print("APLICACIÓN DE TEMPLE SIMULADO")
            plt.plot(np.linspace(0,len(list_energy),len(list_energy)),list_energy)
            plt.xlabel('Iteraciones')
            plt.ylabel('Energía')
            plt.title('TEMPLE SIMULADO')
            plt.show()

            #Ploteamos los datos clasificamos por kmeans y los centroides
            print("CLASIFICACIÓN CON Kmeans")
            self.color=['or','go','ko','yo','ob','or','go','ko','yo','ob','ko']
            for i in range(len(self.x_T)):
                plt.plot(self.x_T[i],self.y_T[i],self.color[self.cluster[i]])

            colorc=['xr','xg','xk','xy','xr','xg','xk','xy','xg']
            for i in range(kn):    
                plt.plot(self.x_cluster[i],self.y_cluster[i],colorc[i])
            plt.title('CLASIFICACIÓN CON Kmeans')
            plt.xlabel('Color')
            plt.ylabel('Relación área círculo')
            plt.show()

    def class_pruebas(self,data,tabla,kn=4,foto=False):

        """Clasificamos imagenes de prueba mediante kmeans y knn"""
        
        self.kmeans_init(plot=True,kn=kn)
        for i in data['data_x'].keys():
            s1_list=[]
            s2_list=[]
            b1_list=[]
            b2_list=[]
            for j,l in zip(data['data_x'][i],data['data_y'][i]):
                s1,b1=kmeans_class([j,l],self.x_cluster,self.y_cluster,self.cluster,self.cluster_T,self.x_Tmax,self.y_Tmax,kn=kn)
                s2,b2=knn(5,[j,l],self.x_T,self.y_T,self.cluster_T,self.x_Tmax,self.y_Tmax)
                s1_list.append(s1)
                s2_list.append(s2)
                b1_list.append(b1)
                b2_list.append(b2)

            sep = '|{}|{}|{}|{}|{}|\n'.format('-'*18,'-'*28,  '-'*10,  '-'*18,  '-'*10)
            rep=i+'\n' 
            rep=rep + str('{0}|    Prueba n°     |            Kmeans          |    %     |        Knn       |     %    |\n{0}'.format(sep))
            for n,s11,b11,s22,b22  in zip(range(len(s1_list)),s1_list,b1_list,s2_list,b2_list):
                rep=str(rep)+'| {0:^15d}  | {1:^26s} | {2:^8s} | {3:^16s} | {4:^8s} | \n{5}'.format(n,s11,b11,s22,b22,sep)
            print( str(rep))
        if foto:
            self.dd.plotimage(tabla,6)

    def class_foto(self,name,camera=False):
        
        """Clasificamos una imagen de prueba mediante kmeans y knn"""
        
        if camera:
            self.pr.process(name,foto=False)
        else:
            self.kmeans_init(plot=False)
            path, _ = os.path.split(os.path.abspath(__file__))
            self.pr.process(path+rf'\{name}')
        [x,y,a,a2,a3]=self.pr.data()
        s1,b1=kmeans_class([x,y],self.x_cluster,self.y_cluster,self.cluster,self.cluster_T,self.x_Tmax,self.y_Tmax)
        s2,b2=knn(5,[x,y],self.x_T,self.y_T,self.cluster_T,self.x_Tmax,self.y_Tmax)

        if camera:
            return s1+b1,s2+b2,self.pr.x,self.pr.y,self.pr.w,self.pr.h
        else:
            print('Kmeans:  ',s1,b1)
            print('Knn:  ',s2,b2)
            tabla={'imagen':{'foto':[self.pr.imgT]}}
            self.dd.plotimage(tabla,1)

def kmeans(x_T,y_T,cluster,kn=4,anelling=False):

    #cluster: es la clasificación inicial de los datos de manera aleatoria.
    #Definimos inicialmente los centroides en cero
    x_i=0
    y_i=0

    dT=np.zeros(kn)
    it=True
    ii=0
    while(it):
        
        x_cluster=np.zeros(kn)
        y_cluster=np.zeros(kn)

        #Calculamos la distancia a cada centroide, y clasificamos la información en función de la menor distancia
        for i in range(len(x_T)):

            if ii>0:

                d=np.power((x_T[i]-x_i),2)+np.power((y_T[i]-y_i),2)
                a=d.argmin()
                cluster.append(a)
                dT[a]+=d[a]
                
            x_cluster[cluster[i]]+=x_T[i]
            y_cluster[cluster[i]]+=y_T[i]

        #Calculamos los centroides.
        for i in range(kn):

            x_cluster[i]/=(cluster.count(i)+1e-5)
            y_cluster[i]/=(cluster.count(i)+1e-5)
            dT[i]/=(cluster.count(i)+1e-5)

        #Verificamos si el algoritmo converge
        if np.all(abs(x_cluster-x_i)<0.001) and np.all(abs(y_cluster-y_i)<0.001):

            
            if anelling:
                return dT
            else:
                return cluster,x_cluster,y_cluster
            
        else:

            x_i=x_cluster
            y_i=y_cluster
            ii+=1
            cluster=[]

def kmeans_class(data,x_cluster,y_cluster,cluster,cluster_T,x_Tmax,y_Tmax,kn=4):

    """Clasificamos un dato de entrada en funcion de los centroides calculados con el algoritmo k-means"""

    frutas=['limon','tomate','naranja','banana']

    #Normalizamos los datos
    data[0]=data[0]/x_Tmax
    data[1]=data[1]/y_Tmax
    data=np.array(data)

    #Calculamos la distancia del dato con respecto a los centroides y nos quedamos con el menor.
    d=np.power((data[0]-x_cluster),2)+np.power((data[1]-y_cluster),2)
    a=d.argmin()

    dict1={}
    for i in range(kn):
        dict1.update({f'{i}':0})
    #Calculamos la presición del algoritmo.
    for i in range(len(cluster)):

        if a==cluster[i]:

            dict1[f'{cluster_T[i]}']+=1
    T2=np.array(list(dict1.values()))
    T=np.sum(T2) 

    a=T2.argmax()
    aux=dict1[f'{a}']
    s=frutas[a] 
    b= f'{round(aux/(T+0.001)*100)}%'
    return s,b

def knn(k,data,x_T,y_T,cluster_T,x_Tmax,y_Tmax):

    #Aplicamos el algoritmo K nearest neighbours.
    frutas=['limon','tomate','naranja','banana']

    #Normalizamos los datos
    data[0]=data[0]/x_Tmax
    data[1]=data[1]/y_Tmax
    data=np.array(data)

    #Calculamos la distancia de nuestro dato de entrada con todos los vecinos
    d=np.power((data[0]-x_T),2)+np.power((data[1]-y_T),2)
    aux=list(d.copy())

    #Ordenamos de menor a mayor
    d.sort()
    
    dict1={'0':0,'1':0,'2':0,'3':0}
    #Seleccionamos los k vecinos mas cercanos
    for i in range(k):
        a=aux.index(d[i])
        dict1[f'{cluster_T[a]}']+=1
       
    T2=np.array(list(dict1.values()))
    T=np.sum(T2)   
    #Clasificamos en función de los vecinos

    a=T2.argmax()
    aux=dict1[f'{a}']
    s=frutas[a] 
    b= f'{round(aux/T*100)}%'
    return s,b

