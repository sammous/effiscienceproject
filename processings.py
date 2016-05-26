#-*- coding:utf-8 -*-

"""
Ce module permet de traiter les données obtenues lors de la partie tracking pour les améliorer et corriger les erreurs. Chaque traitement appliqué à la liste de donnée est documenté au niveau du script de la méthode.
"""

import pylab as p
import numpy as np
import horseTracking as hT
import time


class Resultat(object):
    def __init__(self,video,nb_image=3,process_ratio=2,listeTrajSurX=[],listeTrajSurY=[],listeTaillesComposantes=[],listeVitesseSurX=[], listeVitesseSurY=[], listeTemps=[], hauteurDeLaVideo=480, largeurDeLaVideo=640,fps=25):

        # Initialisation des champs de la classe Resultat
        self.nb_image = nb_image
    	self.video=video
    	self.process_ratio=process_ratio
    	self.x = listeTrajSurX
    	self.y = listeTrajSurY
    	self.tC = listeTaillesComposantes
    	self.t = listeTemps
    	self.vx = listeVitesseSurX
    	self.vy = listeVitesseSurY
    	self.h = hauteurDeLaVideo
    	self.w = largeurDeLaVideo
    	self.fps = max(fps,1)


def main(r,display=False):
    """ Fonction où est appelée l'ensemble des fonctions de traitement utilisées dans ce module."""

    # Initiaisation des variables
    cx=list(r.x)
    cy=list(r.y)
    fps=r.fps
    nb_image=r.nb_image
    h=r.h
    w=r.w

    if display==True:
    	print("\n-------------------------")
    	print("On est dans processings.py")

    # Appel des méthodes de traitement
    cx,cy=correctionPositionInitiale(cx,cy,display)
    cx,cy=correctionPointsImprobables(cx,cy,h,w,display)
    cx,cy=correctionSautImages(cx,cy,nb_image,display)
    cx,cy=trajUnidirectionnelle(cx,cy,display)
    cx,cy=correctionRegLineaire(cx,cy,display)
    cx,cy=correctionTracerCourbes(cx,cy,h,display)
    r.vx,r.vy,r.t=calculVitesse(cx,cy,fps,display)

    # Enregistrement des traitements
    r.x=list(cx)
    r.y=list(cy)

    return r




""" Les différentes méthodes utilisées pour le traitement. """

def correctionSautImages(cx,cy,nb_image,display=False):
    """ On ajoute chaque valeur nb_image fois à la liste pour compenser le nombre d'images sautées lors de la partie tracking.
        (nb_image n'ont pas été traités entre deux images utilisées par l'algorithme de tracking, il faut donc les rajouter) """
    cx1=[]
    cy1=[]
    for el in cx:
        for i in range(nb_image):
            cx1.append(el)

    for el in cy:
        for i in range(nb_image):
            cy1.append(el)

    if display==True:
	print("\ncorrectionSautImages")
	print("cx :", len(cx1), cx1)
	print("cy :", len(cy1), cy1)

    return(cx1,cy1)


def correctionPositionInitiale(cx,cy,display=False):
    """ On s'assure que le centre de gravité ne passe pas de (0,0), coordonnée par défaut lorsque que le cheval n'a pas été détécté par l'algo de tracking, à sa première coordonnée en seulement 2 images.
        On aurait sinon des vitesses très grandes, préjudiciables pour l'interprétation des résultats. """
    i=0
    while cx[i]==0:
        i+=1
    cx_deb=cx[i]
    cy_deb=cy[i]
    for j in range(len(cx)):
        if j<i:
            cx[j]=cx_deb
            cy[j]=cy_deb

    if display==True:
    	print("\ncorrectionPositionInitiale")
    	print("cx :", len(cx), cx)
    	print("cy :", len(cy), cy)

    return(cx,cy)


def trajUnidirectionnelle(cx,cy,display=False):
    """ HYPOTHESE FORTE: On suppose que le cheval est dans une course et qu'il n'a aucune raison de faire marche arrière.
	Ainsi, si le centre de gravité se déplace à un moment donné dans le sens inverse du cheval, on suppose qu'il s'agit d'une erreur lors du tracking. """

    if cx[0]<cx[len(cx)-1]:
        for i in range(len(cx)-1):
            if cx[i]>=cx[i+1]:
                cx[i+1]=cx[i]
    else:
        for i in range(len(cx)-1):
            if cx[i]<=cx[i+1]:
                cx[i+1]=cx[i]
    if display==True:
    	print("\ntrajUnidirectionnelle")
    	print("cx :", len(cx), cx)
    	print("cy :", len(cy), cy)

    return(cx,cy)


def correctionRegLineaire(cx,cy,display=False):
    """ Entre des ensembles de points identiques, le déplacement du cheval est supposé linéaire.
        Ex: si cx=[1,1,1,10,10,20,30,42,42,50,60], alors après traitement cx=[1, 4, 7, 10, 15, 20, 30, 42, 46, 50, 60]  """

    i=0
    while i<len(cx)-1:
        j=i
        while cx[i]==cx[j]:
            j+=1
            if j>=len(cx):
                j=len(cx)-1
                break
        k=j-i

        for n in range(k):
            cx[i+n]=cx[i+n]+n*(cx[j]-cx[i])/k
        i=j
        if j==len(cx)-1:
            i+=1

    if display==True:
	print("\ncorrectionRegLineaire")
	print("cx :", len(cx), cx)
	print("cy :", len(cy), cy)

    return(cx,cy)


def correctionPointsImprobables(cx,cy,h,w,display=False):
    """ Hypothèse: Le cheval ne peut se téléporter. => Si un point est trop éloigné de son prédécesseur ou de son successeur, on peut le considérer comme une erreur. """
    cy.reverse()
    i=0
    while (i<=len(cy)-2):
	if abs(cy[i+1]-cy[i])>(h/7.0):
	    cy[i]=cy[i+1]
	    cx[i]=cx[i+1]
	else:
	    i+=1
    cy.reverse()
    i=0
    while (i<=len(cy)-2):
	if abs(cy[i+1]-cy[i])>(h/7.0):
	    cy[i+1]=cy[i]
	    cx[i+1]=cx[i]
	else:
	    i+=1

    if display==True:
	print("\ncorrectionPointsImprobables")
	print("cx :", len(cx), cx)
	print("cy :", len(cy), cy)
    return cx,cy


def correctionTracerCourbes(cx,cy,h,display=False):
    """ Les coordonnées d'une image ont pour origine le coin en haut à gauche, alors qu'une courbe a pour origine le coin en bas à gauche. Il faut donc appliquer une modification d'origine. """
    cy_new=[]
    for el in cy:
        cy_new.append(h-el)

    if display==True:
	print("\ncorrectionTracerCourbes")
	print("cx :", len(cx), cx)
	print("cy :", len(cy), cy_new)
    return cx,cy_new


def calculVitesse(cx,cy,fps,display=False):
    """ Calcul de la vitesse du cheval (en px/s) grâce aux coordonnées du centre de gravité"""

    vx = []
    vy = []
    t  = []

    for i in range(1, len(cx)):
	    vx.append((cx[i]-cx[i-1])*fps)
	    vy.append((cy[i]-cy[i-1])*fps)
	    t.append(i/fps)

    if display==True:
	print("\ncalculVitesse")
	print("vx :", len(vx), vx)
	print("vy :", len(vy), vy)
	print("t :", len(t), t)
    return [vx,vy,t]


if __name__ == '__main__':
    r = Resultat("video/base9.avi",3)
    r=hT.main(r)
    debut=time.time()
    status = main(r,True)
    fin=time.time()
    print("\n")
    print("Temps necessaire pour le processing: "+str(fin-debut)+" seconde(s)")
