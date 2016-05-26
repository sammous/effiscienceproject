#-*- coding:utf-8 -*-

"""
Ce module a pour but de détecter et tracker le centre de gravité du cheval et du cavalier en mouvement. Le principe utilisé est celui de la soustraction d'images succéssives (les pixels qui ont été modifiés entre deux images fixes successives ont une forte probabilité d'appartenir au cheval en mouvement) . On soustrait ainsi à chaque étape l'image n° [i+nb_image] par l'image n° [i] de la vidéo. Le paramètre nb_image permet ainsi d'augmenter ou de diminuer la distance effectuée par le cheval entre deux images traitées, et donc de détecter le mouvement plus ou moins facilement. (Plus nb_image est grand, plus le cheval s'est déplacé et donc plus sa détection est facile. Cependant, si nb_image est trop grand, le mouvement détecté n'est alors plus précis.-

Différents traitements sont ensuite appliqués (filtre Blur, morphologies mathématiques -ouverture et gradient-, seuillage...) pour diminuer le bruit de l'image obtenue précédemment. Des fonctions de la bibliothèque scipy permettent notamment d'isoler la plus grosse composante connexe de l'image et d'en détecter le centre de gravité à chaque instant.
"""

import cv2
import cv2.cv as cv
import numpy as np
import processings as pr
from scipy import ndimage
import math
import time



varBlur=33	    # Intensité de la fonction blur
minThreshold=15     # Valeur min du threshold
maxThreshold=255    # Valeur max du threshold
kerOpen=21          # Intensité de l'ouverture morphologique
kerGradient=6       # Intensité du gradient morphologique

"""Variables globales utilisées pour le tracking"""
"""
varBlur=48	    # Intensité de la fonction blur
minThreshold=21     # Valeur min du threshold
maxThreshold=255    # Valeur max du threshold
kerOpen=33          # Intensité de l'ouverture morphologique
kerGradient=6       # Intensité du gradient morphologique
"""

"""Définition de la classe HorseTracking"""

class HorseTracking(object):

    def __init__(self, video, nb_image, process_ratio):
        """Initialisation de la capture, du stockage des premières images et des premiers traitements"""

        self.process_ratio=process_ratio
        self.nb_image=nb_image

        # Initialisation de la capture vidéo et de la première frame
        self._capture = cv2.VideoCapture(video)
        if self._capture is None:
            raise Exception("Aucune caméra détectée, ou vidéo introuvable")
        self._frame1 = self.get_frame()

        # Initialisation du redimensionnement des images issues de la capture
        self._height = self._frame1.shape[0] / self.process_ratio
        self._width = self._frame1.shape[1] / self.process_ratio
        self._resolution = self._height * self._width

        # Création de l'ensemble des matrices/images utiles pour la suite
        self._last_frame = np.zeros(self._frame1.shape, np.uint8)
        self._frame2_grayscale = np.zeros((self._height, self._width), np.uint8)
        self._frame_resized = np.zeros((self._height, self._width), np.uint8)
        self._diff = np.zeros((self._height, self._width), np.uint8)

        # Redimensionnement et convertissement de la première image de la capture en niveau de gris
        self._frame_resized = cv2.resize(self._frame1, (self._width, self._height))
        self._frame1_grayscale = cv2.cvtColor(self._frame_resized, cv2.COLOR_RGB2GRAY)


    def get_frame(self):
        """Retourne la dernière frame capturée en prenant en compte le nombre d'image nb_image sautée à chaque appel de la fonction"""

        for i in range(self.nb_image):
            ret, frame = self._capture.read()
            print ret
        return frame


    def run(self):
        """Retourne la dernière frame capturée en prenant en compte le nombre d'image nb_image sautée à chaque appel de la fonction"""

        frame = self.get_frame()

        if frame is None:
            return True

        else:

            # Copie de la frame obtenue par get_frame(),redimensionnement et conversion de celle-ci en niveau de gris
            self._last_frame = np.copy(frame)
            self._frame_resized = cv2.resize(frame, (self._width, self._height))
            self._frame2_grayscale = cv2.cvtColor(self._frame_resized, cv2.COLOR_RGB2GRAY)

            # Différence entre les deux dernières frames, et traitement de celle-ci avec filtre Blur et un threshold
            self._diff = cv2.absdiff(self._frame1_grayscale, self._frame2_grayscale)
            self._diff = cv2.blur(self._diff, (varBlur,varBlur))
            _, self._diff = cv2.threshold(self._diff, minThreshold, maxThreshold, cv2.THRESH_BINARY_INV)

            # Stockage de la frame n°2 dans la frame n°1
            self._frame1_grayscale = np.copy(self._frame2_grayscale)

        return False




""" Définition de la fonction main()"""

def main(r,display=False):

    # Initialisation des variables
    cx=0
    cy=0
    liste_cx=[]
    liste_cy=[]
    liste_time=[]
    liste_tailles=[]

    # Initialisation des constantes
    kernelOpen = np.ones((kerOpen,kerOpen),np.uint8)
    kernelGrad = np.ones((kerGradient,kerGradient),np.uint8)
    s=[[1,1,1],[1,1,1],[1,1,1]]
    video=r.video
    nb_image=r.nb_image
    process_ratio=r.process_ratio

    #Création de l'objet HorseTracking, et obtention du fps de la vidéo (frame par seconde = temps entre deux images succéssives)
    capture = HorseTracking(video,nb_image,process_ratio)
    fps = max(1,capture._capture.get(cv2.cv.CV_CAP_PROP_FPS))

    while True:
        debut=time.time()
        if capture.run() ==True:
            break

        # Utilisation de la morphologie mathématique (ouverture et gradient) sur l'image de la différence.
        diffOpened = cv2.morphologyEx(capture._diff, cv2.MORPH_OPEN, kernelOpen)
        diffAfterGradient = cv2.morphologyEx(diffOpened, cv2.MORPH_GRADIENT, kernelGrad)

        # Utilisation de la bibliothèque Scipy pour déterminer le plus grosse composante connexe.
        lab,nb_objets=ndimage.label(diffAfterGradient,s)
        tailles=ndimage.sum(diffAfterGradient,lab,range(nb_objets+1))
        masque=tailles<max(tailles)/3   # On écarte toutes les composantes dont l'aire est inférieure à 1/3 de l'aire maximale de l'image précédente.
        trop_petites=masque[lab]
        lab2=lab.copy()
        lab2[trop_petites]=0

        # Calcul du centre de gravité de la plus grosse composante connexe détéctée.
        centroid=ndimage.measurements.center_of_mass(diffAfterGradient,lab2,xrange(1,nb_objets+1))
        for i in range(nb_objets):
                if math.isnan(centroid[i][0])==False:
                    cx=centroid[i][1]*capture.process_ratio
                    cy=centroid[i][0]*capture.process_ratio

        # Stockage de toutes les informations utiles dans les listes
        liste_cx.append(cx)
        liste_cy.append(cy)
        liste_tailles.append(max(tailles))

        # Affichage de la vidéo si souhaité.
        if display==True:
            cv2.circle(capture._last_frame,(int(cx),int(cy)),2,(0,0,255),3)

            cv2.imshow('Video de base', capture._last_frame)
            cv2.imshow('Video ant-processing', capture._diff)
            cv2.imshow('Video post-processing', diffAfterGradient)
        fin=time.time()
        temps= fin - debut
        k = cv2.waitKey(int(r.process_ratio*1000/max(fps,1)-temps)) & 0xFF
        if k == 27:
            break

    # Desctruction de toutes les fenêtres
    cv2.destroyAllWindows()

    r.x=liste_cx
    r.y=liste_cy
    r.tC=liste_tailles
    r.fps=fps
    r.h=capture._frame1.shape[0]
    r.w=capture._frame1.shape[1]


    if display==True:
        print("\n-------------------------")
        print("On est dans horseTracking.py")
        print("cx: ", len(r.x), r.x)
        print("cy: ", len(r.y), r.y)
        print("tailles des composantes: ", len(r.tC), r.tC)
        print("fps: ", r.fps)
        print("h: ", r.h)
        print("w: ", r.w)

    return r


if __name__ == '__main__':
    r = pr.Resultat("video/base.avi",3)
    debut=time.time()
    status = main(r,True)
    fin=time.time()
    print(fin-debut)
