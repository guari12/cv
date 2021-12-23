
import os
import json
from imageprocessing import processimage

class Cache():
    
    def __init__(self,path,archivo=True,foto=False):

        self.path=path
        self.archivo=archivo
        self.Foto=foto
        try:
            if archivo:
                tf = open(self.path+"/cache.json", "r")
                self.data = json.load(tf)
                self.tabla={}
                tf.close()
            else:
                self.tabla={}
                self.data = {}

        except :
            self.tabla={}
            self.data = {}

        self.frutas=['limon','tomate','naranja','banana']
        
    def crear(self,name,imagen=False):

        """Procesa las imagenes y crea un archivo .json para guardar los parámetros"""

        pr=processimage()
        contenido = os.listdir(self.path+rf'\{name}')
        data_fruta_x_dict={}
        data_fruta_y_dict={}
        img_dict={}
        img=[]
        data_fruta_x=[]
        data_fruta_y=[]
        rel_nu=[]
        rel_ret=[]
        rel_per=[]

        if len(self.tabla) != 0 and imagen:

            img_dict=self.tabla['imagen']       

        if len(self.data) != 0:
            
            data_fruta_x_dict=self.data['data_x']
            data_fruta_y_dict=self.data['data_y']
            rel_nu=self.data['nu02']
            rel_ret=self.data['ret']
            rel_per=self.data['per']

        if self.Foto:

            pr.process(self.path)
            x,y,f,g,h=pr.data()

            if imagen:
                img.append(pr.imgT)

            data_fruta_x.append(x)
            data_fruta_y.append(y)
            rel_nu.append(h)
            rel_per.append(f)
            rel_ret.append(g)

        for i in contenido:
            pr.process(self.path+rf'\{name}\{i}')
            x,y,f,g,h=pr.data()

            if imagen:
                img.append(pr.imgT)

            data_fruta_x.append(x)
            data_fruta_y.append(y)
            rel_nu.append(h)
            rel_per.append(f)
            rel_ret.append(g)

        if imagen:
            img_dict.update({name:img})

        data_fruta_x_dict.update({name:data_fruta_x})
        data_fruta_y_dict.update({name:data_fruta_y})

        if imagen:
            self.tabla.update({'imagen':img_dict})

        self.data.update({'data_x':data_fruta_x_dict})
        self.data.update({'data_y':data_fruta_y_dict})
        self.data['nu02']=rel_nu
        self.data['ret']=rel_ret
        self.data['per']=rel_per

        if self.archivo:
            tf = open(self.path+"/cache.json", "w")
            json.dump(self.data,tf)
            tf.close()

    def leer_imagenes(self):
        
        """Leamos la imagenes de la base de datos"""

        for i in self.frutas:
            self.crear(i,imagen= True)

def path_dir():
    path, _ = os.path.split(os.path.abspath(__file__))
    return path