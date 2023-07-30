
from cmd import Cmd
import cv2 as cv
from imageprocessing import data_plot,clasificacion
from simulated_anneling import anneling, ley_enfriamiento
from cache import Cache,path_dir

class Consola(Cmd):

    panelesCreados = 0
    doc_header = 'Comandos documentados'

    def __init__(self):

        Cmd.__init__(self)
        path=path_dir()
        self.c=Cache(path)
        T=ley_enfriamiento(1000,700,1.05)
        temple=anneling(T)
        self.cls=clasificacion(temple)
        self.dd=data_plot()
        try:
            self.cls.data_accond(self.c.data)
        except:
            self.c.leer_imagenes()
            self.cls.data_accond(self.c.data)

    def do_Prueba(self,args):

        """Realiza la clasificación de los datos de prueba con los algoritmos K-means y Knn. Debe pasarse el numero de cluster, por defecto es 4"""

        path=path_dir()
        path=path+rf'\prueba'
        c2=Cache(path)
        c2.leer_imagenes()
        if args!='':
            self.cls.class_pruebas(c2.data,c2.tabla,kn=int(args))
        else:
            self.cls.class_pruebas(c2.data,c2.tabla,kn=4,foto=True)

    def do_leerbd(self,args):

        """Lee la base de datos"""

        self.c.leer_imagenes()
        self.cls.data_accond(self.c.data)

    def do_plotbd(self,args):

        """Plotea la base de datos"""

        self.c.leer_imagenes()
        self.cls.data_accond(self.c.data)
        self.dd.plotimage(self.c.tabla,5)

    def do_plotdatos(self,args):

        """Plotea parámetros caraterísticos de la base de datos"""

        self.dd.plotdata(self.c.data,self.cls)

    def do_foto(self,args):

        """Clasifica una foto dada con los algoritmos K-means y Knn"""

        self.cls.class_foto(args)

    def do_camera(self,args):

        """"Clasificación de frutas mediante la camera"""
        self.cls.kmeans_init(plot=True,kn=9)
        cap = cv.VideoCapture(1)
        if not cap.isOpened:
            print('--(!)Error opening video capture')
            exit(0)
        while True:
            
            ret, frame = cap.read()
            fr=frame.copy()
            d,b,x,y,w,h=self.cls.class_foto(frame,camera=True)
            text_=f'K-means:{d}; Knn:{b}'
            cv.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(fr,text_,(x,y-10),0,0.5,(0,255,0),1)
            cv.imshow('Clasificador de frutas', fr)
            
            if frame is None:
                print('--(!) No captured frame -- Break!')
                break
            if cv.waitKey(50) >=0:
                break

    def default(self, args):

        self.bool=0
        print("Error. El comando \'" + args + "\' no esta disponible")

    def precmd(self, args):

        print("--------------------------------------------------------")
        return(args)

    def do_exit(self,args):

        """"Terminar consola"""

        raise SystemExit
