#-*- coding:utf-8 -*-

""" Ce module permet d'afficher le résultat des modules horseTracking.py et processings.py sur la vidéo. Nous extrayons également un petit carré d'image contenant uniquement le cheval. Ce carré sera utilisé pour le développement et la modélisation de l'algorithme de modèle articulé 2D que nous essayons actuellement de mettre en place.

Pour finir, ce module permet également d'afficher l'ensemble des résultats sur un ensemble de courbes (trajectoire, vitesse du cheval...) """

import cv2.cv as cv
import cv2
import time
import numpy as np
import pylab as p
import detectModel2D as d2D



def main(r,taille,ombre,direction,process_ratio):
    affichageVideo(r,taille,ombre,direction,process_ratio)
    plotGlobal(r)    
    return r

def affichageVideo(res,taille,ombre,direction,process_ratio):
    """Affichage des résultats sur la vidéo et extraction d'un petit carré d'image contenant le cheval. Ce carré sera utilisé pour le développement et la modélisation de l'algorithme de model articulé 2D que nous essayons actuellement de mettre en place."""
    
    # Initialisation des variables de la méthode
    video=res.video; liste_cx=list(res.x); liste_cy=list(res.y);
    nb_image=res.nb_image; h=res.h; w=res.w; lt=res.tC; 
    
    # Initialisation de la lecture de la video avec le bibliothèque OpenCV
    cap = cv2.VideoCapture(video)
    fps = max(cap.get(cv2.cv.CV_CAP_PROP_FPS),1)
    for i in range(int(nb_image)):
	_, frame = cap.read()
    last_frame=frame
    first_frame=frame
    
    # Corriger le changement d'origine sur une image (en haut à gauche) et une courbe (en bas à gauche) en python pour que le entre de gravité apparaisse bien sur le cheval.
    for i in range(len(liste_cy)):
	liste_cy[i]= h-liste_cy[i]
    
    i=0
    j=0
    liste_prev=[[0,0],[0,0],[0,0]]
    fourcc = cv2.cv.CV_FOURCC('I','4','2','0')
    out = cv2.VideoWriter('output3.avi',fourcc, int(res.fps/3), (res.w,res.h))  
    while True:
	debut=time.time()    
	if frame==None:
	    break	
	
	if i<len(liste_cx):
	    cx=liste_cx[i]
	    cy=liste_cy[i]
	    i+=1
	
	# Détermination des valeurs importantes pour isoler le ROI (Region Of Interest)
	a=max(int(max(lt)/20000.0),w/6)
	b=max(int(max(lt)/20000.0),h/4)
	roi_xmin=max(1,int(cy-b))
	roi_xmax=min(h,int(cy+b))
	roi_ymin=max(0,int(cx-a))
	roi_ymax=min(w,int(cx+a))
	 
	
	
	# Calcul du ROI (Region Of Interest)
	if ombre:
	    last_roi=last_frame[roi_xmin:roi_xmax,roi_ymin :roi_ymax]
	else: 
	    last_roi=first_frame[roi_xmin:roi_xmax,roi_ymin :roi_ymax]
	roi = frame[roi_xmin:roi_xmax,roi_ymin :roi_ymax]
	roi_diff=cv2.absdiff(roi,last_roi)
	
	# Traitement au ROI (Region Of Interest)

	"""
        roi_gray=gray = cv2.cvtColor(roi_diff, cv2.COLOR_BGR2GRAY)
	roi_blured = cv2.blur(roi_gray, (24,24))
	_, roi_threshold = cv2.threshold(roi_blured,40, 255, cv2.THRESH_BINARY_INV)	    
	kernel = np.ones((18,18),np.uint8)
	kernel1 = np.ones((7,7),np.uint8)
	roi_opened = cv2.morphologyEx(roi_threshold, cv2.MORPH_OPEN, kernel)
        """
	
	roi_gray= cv2.cvtColor(roi_diff, cv2.COLOR_BGR2GRAY)
	roi_blured = cv2.blur(roi_gray, (19,19))
	_, roi_threshold = cv2.threshold(roi_blured,19, 255, cv2.THRESH_BINARY_INV)	    
	kernel = np.ones((25,25),np.uint8)
	roi_opened = cv2.morphologyEx(roi_threshold, cv2.MORPH_OPEN, kernel)
	roi_final = roi_opened
	
	liste=liste_prev
	r_model2D = [ 0, 0, 0, 0, 0, 0, 0]
	r_model2D, liste_prev = d2D.centerPointsDetermination(roi_final,taille,process_ratio,r_model2D,direction)
	

	
	      
	
	
		
	
	# Affichage de la vidéo
	if frame!=None:
	    
	    j+=1
	    frame_bis=np.copy(frame)
	    cv2.circle(frame_bis,(int(cx),int(cy)),2,(0,0,255),3)
	    cv2.rectangle(frame_bis, (int(cx-a),int(cy+b)),(int(cx+a),int(cy-b)), 3)
	    cv2.circle(frame_bis,(int(cx+liste[0][0]-a),int(cy+liste[0][1]-b)),2,(0,255,0),3)
	    cv2.circle(frame_bis,(int(cx+liste[1][0]-a),int(cy+liste[1][1]-b)),2,(0,255,0),3)
	    cv2.circle(frame_bis,(int(cx+liste[2][0]-a),int(cy+liste[2][1]-b)),2,(0,255,0),3)
	    print(liste)
	    cv2.imshow('result',frame_bis)	
	    cv2.imshow('res',roi_final)
	    
	    
	    #out.write(frame_bis)
        
        # Changement de frame pour le prochain tour de la boucle
	last_frame=frame
	_, frame = cap.read()	

	# Paramétrer la sortie de la vidéo en appuyant sur la touche échap
	fin=time.time()
	temps=fin-debut		
	k = cv2.waitKey(int(1000/fps-temps)) & 0xFF
	if k == 27:
	    break	
    cv2.destroyAllWindows()















""" Fonctions de tracé de courbes. """

def smooth(x, win=7):
    """ Les résultats obtenus dans la partie traitement.py comportent souvent énormément de bruits, et particulièrement les valeurs de vitesse. La fonction smooth permet de lisser les valeurs obtenues et d'ainsi réduire le bruit."""
    s = np.r_[x[win-1:0:-1],x,x[-1:-win:-1]]
    w = np.ones(win, 'd')
    y = np.convolve(w/w.sum(),s,mode='valid')
    return y[(win-1)/2:-(win-1)/2]

def plotSimple(liste_abs,liste_ord,varSmooth,couleur='r',titre='',axe_x='',axe_y=''):
    """ Permet de créer directement une courbe en rentrant les paramètres nécessaires (listes de valeur, nom des axes...)."""
    if len(liste_abs)>len(liste_ord):
        liste_abs=liste_abs[:len(liste_ord)]
    if len(liste_abs)<len(liste_ord):
        liste_ord=liste_ord[:len(liste_abs)]
    p.figure()  
    p.plot(liste_abs,smooth(liste_ord,varSmooth),couleur)
    p.xlabel(axe_x)
    p.ylabel(axe_y)
    p.title(titre)
    
def plotGlobal(r):
    """ Trace automatiquement 5 courbes très intéressantes pour l'analyse de la course du cheval -1)cy=f(cx) -2)cx=f(t) -3)cy=f(t) -4)vx=f(x) -5)vy=f(y)"""
    plotSimple(r.x,r.y,4,'r','Trajectoire','x en px','y en px')
    p.axis([0.0,r.w,0.0,r.h])
    plotSimple(r.t,r.x,4,'r','x en fonction de t','t en s','x en px')
    p.axis([0.0,max(r.t),0.0,r.w])
    plotSimple(r.t,r.y,4,'r','y en fonction de t','t en s','y en px')
    p.axis([0.0,max(r.t),0.0,r.h])
    plotSimple(r.x,r.vx,15,'b','vx en fonction de x','x en px','vx en px/s')
    plotSimple(r.x,r.vy,15,'b','vy en fonction de x','x en px','vy en px/s')
    plotSimple(r.t,r.vx,15,'b','vx en fonction de t','t en s','vx en px/s')
    plotSimple(r.t,r.vy,15,'b','vy en fonction de t','t en s','vy en px/s')    

from math import *
def modeleArticule(img,cx,cy,w1,t1,t2):
    r1=0.25
    a1=0.5
    a2=2
    t1=radians(t1)
    w2=r1*w1
    h1=a1*w1
    h2=a2*w2
    cy-=h1/2
    pts=np.array([[int(cx-w1/2.0*cos(t1)),int(cy+w1/2*sin(t1))],[int(cx-w1/2.0*cos(t1)+h1*sin(t1)),int(cy+h1+w1/2)],[int(cx+w1/2),int(cy+h1)],[int(cx+w1/2.0),int(cy)]], np.int32)
    #pts = pts.reshape((-1,1,2))
    #pts1=np.array([[int(cx+(w1*cos(t1))/2.0),int(cy+(w1*sin(t1))/2.0)],[int(cx-w1/2*cos(t1)+h1*sin(t1)),int(cy+h1+w1/2*sin(t1))],[int(cx-w1/2*cos(t1)),int(cy+h1+w1/2*sin(t1))],[int(cx+w1/2*cos(t1)),int(cy-w1/2*sin(t1))]],np.int32)
    cv2.polylines(img,[pts],True,(0,255,255))
    return img
    
    
def changementOrigine(x1,y1,xo,yo,x0,y0):
    x2=x1+(x0-xo)
    y2=y1+(X0-xo)
    return x2, y2

def rotation():
    return 0
#cv2.rectangle(img, (int(cx+(w1*cos(t1))/2.0),int(cy+(w1*sin(t1))/2.0)),(int(cx-w1/2*cos(t1)),int(cy+h1+w1/2*sin(t1))), 3)
    #cv2.rectangle(img, (int(cx+w2/2),int(cy)),(int(cx-w2/2),int(cy-h2)), 3)    