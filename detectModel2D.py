#-*- coding:utf-8 -*-

import cv2
import numpy as np
import math
import time

global g_direction
 

i=0
def centerPointsDetermination(img,taille,process_ratio,res,direction):
    global g_direction
    g_direction = direction
    debut=time.time()
    
    
    # Lecture de l'image  
    #img = cv2.imread(name_img,cv2.CV_LOAD_IMAGE_COLOR)
    height, width = img.shape[:2]
    
    
    # Reduction de la taille de l'image sur laquelle on travaille pour gagner en temps de calcul
    height_resize = height / process_ratio
    width_resize = width / process_ratio    
    img_resize = cv2.resize(img, (width_resize, height_resize))
    
    
    # Initialisation des variables
    if direction =='DG':
	x_init=width_resize/2 + width_resize/9
	
    elif direction=='GD':
	x_init=width_resize/2 - width_resize/9
	
    y_init=height_resize/2 + height_resize/9
    
    
    liste_1=[taille/20, taille/30, 0, -taille/30,-taille/20,]
    liste_2=[taille/20,taille/30, 0, -taille/30,-taille/20]
    liste_3=[taille/40,taille/30, 0, -taille/30,-taille/40]
    liste_4=[7,0,-7]   
	
    taille=taille/process_ratio
    
    liste_horse_x=[ int(el/process_ratio)+res[1] for el in liste_1] 
    liste_horse_y=[ int(el/process_ratio)+res[0] for el in liste_3 ] 
    liste_horse_O=[ el+res[2] for el in liste_4]     
    liste_knight_x=[ int(el/process_ratio)+res[3] for el in liste_2]    
    liste_knight_O=[ el+res[4] for el in liste_4 if el+res[4]>-50 and el+res[4]<40]    
    liste_head_y=[ int(el/process_ratio)+res[5] for el in liste_3]  
    liste_head_O=[ el+res[6] for el in liste_4 if el+res[6]>-40 and el+res[6]<40]  
    
    """
    liste_horse_x=[ int(el/process_ratio) for el in liste_1] 
    liste_horse_y=[ int(el/process_ratio) for el in liste_2 ] 
    liste_horse_O=[ el for el in liste_3]     
    liste_knight_x=[ int(el/process_ratio) for el in liste_2]    
    liste_knight_O=[ el for el in liste_4 if el+res[4]>-50 and el+res[4]<40]    
    liste_head_y=[ int(el/process_ratio) for el in liste_2]  
    liste_head_O=[ el for el in liste_4 if el+res[6]>-40 and el+res[6]<40]      
    """
    
    liste_max_pix=[]
    nb=0
    nb_max=0
    for y in liste_horse_y:
	horse=calculHorse(x_init ,y_init + y,taille,0)
	nb=nbrePixels(img_resize,horse)
		
	if nb>=nb_max:
	    nb_max=nb
	    y_max=y   
    
    nb=0
    nb_max=0
    for x in liste_horse_x:
	horse=calculHorse(x_init + x , y_init + y_max,taille,0)
	nb=nbrePixels(img_resize,horse)
	    
	if nb>=nb_max:
	    nb_max=nb
	    x_max=x   
		
    nb=0
    nb_max=0
    for i in liste_horse_O:
	horse=calculHorse(x_init + x_max , y_init + y_max,taille,i)
	nb=nbrePixels(img_resize,horse)
	
	if nb>=nb_max:
	    nb_max=nb
	    horse_max=horse
	    O1_max=i
	
    liste_max_pix.append(nb_max)
    nb=0
    nb_max=0
    for x_knight in  liste_knight_x:
	knight=calculKnight(taille,0,horse_max,x_knight)
	nb=nbrePixels(img_resize,knight)
	    
	if nb>=nb_max:
	    nb_max=nb
	    x_knight_max=x_knight
	    
    nb=0
    nb_max=0
    for j in  liste_knight_O:
	knight=calculKnight(taille,j,horse_max,x_knight_max)
	nb=nbrePixels(img_resize,knight)
	    
	if nb>=nb_max:
	    nb_max=nb
	    knight_max=knight
	    O2_max=j  
    
    
    liste_max_pix.append(nb_max)
    nb=0
    nb_max=0
    if direction=='DG':
	angle_head=-90
    elif direction=='GD':
	angle_head=90    
	
	
    for y_head in  liste_head_y:
	head=calculHead(taille,angle_head,horse_max,knight_max,y_head)
	nb=nbrePixels(img_resize,head)
	    
	if nb>=nb_max:
	    nb_max=nb
	    y_head_max=y_head    
    
    nb=0
    nb_max=0
    
	
    for k in  liste_head_O:
	head=calculHead(taille,k+angle_head,horse_max,knight_max,y_head_max)
	nb=nbrePixels(img_resize,head)
	    
	if nb>=nb_max:
	    nb_max=nb
	    head_max=head
	    O3_max=k  
    liste_max_pix.append(nb_max)	
    
    liste=drawShape(process_ratio*horse_max,process_ratio*knight_max, process_ratio*head_max,img,[el*process_ratio**2 for el in liste_max_pix])
    
    res=[y_max, x_max, O1_max, x_knight_max, O2_max, y_head_max, O3_max]
     
    
    fin=time.time()
    temps=fin-debut
    print(temps)        
    return res, liste


def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point


def rotateRectangle(arrayPt,angle,point):
    newArray = np.zeros((4,2), dtype=int)
    
    for i in range(4):
        temp_point = rotatePoint((point[0],point[1]),(arrayPt[i][0],arrayPt[i][1]),angle)
        newArray[i][0] = temp_point[0]
        newArray[i][1] = temp_point[1]
	
    return newArray

def drawRectangle(point,angle,width,height):
    global g_direction
    array_point = np.zeros((4,2), dtype=int)
    
    if g_direction == 'DG':
	#Point en haut à gauche du rectangle
	array_point[3] = [point[0] - width/2 ,point[1] - height]
	#Point en bas à gauche du rectangle
	array_point[2] = [point[0] - width/2,point[1]]
	#Point en bas à droite du rectangle
	array_point[1] = [point[0] + width/2,point[1]]
	#Point en haut à droite du rectangle
	array_point[0] = [point[0] + width/2, point[1] - height]
	
    elif g_direction == 'GD':
	#Point en haut à gauche du rectangle
        array_point[0] = [point[0] - width/2 ,point[1] - height]
	#Point en bas à gauche du rectangle
	array_point[1] = [point[0] - width/2,point[1]]
	#Point en bas à droite du rectangle
	array_point[2] = [point[0] + width/2,point[1]]
	#Point en haut à droite du rectangle
	array_point[3] = [point[0] + width/2, point[1] - height]	
	
    rotated = rotateRectangle(array_point,angle,point)	
    return rotated
	


def calculHorse(x1,y1,w1,O1):

        # Initialisation des paramètres de rapport
        r1=0.50
		
	# Calcul du rectangle pour le tronc du cheval
	h1=r1*w1	
	horse = drawRectangle([x1,y1],O1,w1,h1)
		
	return horse


def calculKnight(w1,O2,horse,x_knight):

        # Initialisation des paramètres de rapport
        r2=1.6
	a2 = 0.32
		
	#Calcul de rectangle pour le cavalier
	w2=a2*w1
	h2=r2*w2
	#equation affine
	x2= (horse[0][0]*7/16 + horse[3][0]*9/16) 
	y2= (horse[0][1]*7/16 + horse[3][1]*9/16)
	
	a=0
	if float(horse[0][0]-horse[3][0])!=0:
	    a = float(horse[0][1]-horse[3][1])/float(horse[0][0]-horse[3][0])
	b = horse[0][1] - a * horse[0][0]
	x2 += x_knight
	y2 = a * x2 + b
		
	knight = drawRectangle([x2,y2],O2,w2,h2)
	
	return knight
    
    
def calculHead(w1,O3,horse,knight,y_head):

        # Initialisation des paramètres de rapport
        r3=1.25
	a3= 0.35
		
	#Calcul de rectangle pour le cavalier
	w3=a3*w1
	h3=r3*w3
	#equation affine
	x3= horse[3][0]*6/7 + horse[2][0]/7
	y3= horse[3][1]*6/7 + horse[2][1]/7
	a=0
	if float(horse[3][0]-horse[2][0])!=0:
	    a = float(horse[3][1]-horse[2][1])/float(horse[3][0]-horse[2][0])
	    b = horse[3][1] - a * horse[3][0]
	    y3 += y_head
	    y3 = a * x3 + b	
	    x3= (y3 - b) / a
	
		
	head = drawRectangle([x3,y3],O3,w3,h3)
	
	return head
    
    
def drawShape(horse, knight, head, img,liste_max):
        #img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        center_horse = np.zeros((1,2), dtype=int)
	center_knight = np.zeros((1,2), dtype=int)
	center_head = np.zeros((1,2), dtype=int)
	
	center_horse[0]=[(horse[1][0]+horse[3][0])/2,(horse[1][1]+horse[3][1])/2]
	center_knight[0]=[(knight[1][0]+knight[3][0])/2,(knight[1][1]+knight[3][1])/2]
	center_head[0]=[(head[1][0]+head[3][0])/2,(head[1][1]+head[3][1])/2]
        a=False
	if showRectangle(img, horse,liste_max[0],0.7):
	    cv2.polylines(img,[center_horse],True,(0,0,255),5)		    
	    cv2.polylines(img, [horse], True, (0, 255, 0), 2)
	    a=True
	else:
	    center_horse[0]=[-10000,-10000]

	if showRectangle(img, knight,liste_max[1],0.4) and a==True:
	    cv2.polylines(img,[center_knight],True,(0,0,255),5)		    
	    cv2.polylines(img, [knight], True, (0, 255, 0), 2)
	else:
	    center_knight[0]=[-10000,-10000]
	

	if showRectangle(img, head,liste_max[2],-1) and a==True:
	    cv2.polylines(img,[center_head],True,(0,0,255),5)		    
	    cv2.polylines(img, [head], True, (0, 255, 0), 2)
	else:
	    center_head[0]=[-10000,-10000]
	
	#cv2.imwrite("out_roi/out_roi"+str(i)+".jpg",img)
	#i+=1
	return [center_horse[0],center_knight[0],center_head[0]]
    
def showRectangle(img, rectangle,nb_max,rapport):
    
    dist_x = math.sqrt((rectangle[2][0] - rectangle[1][0]) ** 2 + (rectangle[2][1] - rectangle[1][1]) ** 2)
    dist_y = math.sqrt((rectangle[0][0] - rectangle[1][0]) ** 2 + (rectangle[0][1] - rectangle[1][1]) ** 2)
    pixel_in_rectangle = dist_x * dist_y
    return nb_max > rapport * pixel_in_rectangle
    
def nbrePixels(img, arrayPt):
	def sign(x):
		if x <= 0: return -1
		return 1
	count = 0
	height, width = img.shape[:2] 
	a = np.transpose(arrayPt)[0]
	b = np.transpose(arrayPt)[1]
	b=[height-b[0],height-b[1],height-b[2],height-b[3]]
	coef1 = 0; coef2 = 0; coef3 = 0; coef4 = 0;
	
	if float(a[0] - a[3])!=0 and float(a[3] - a[2])!=0 and float(a[2] - a[1])!=0 and float(a[1] - a[0])!=0: # rectangle non droit
		coef1 = float(b[0] - b[3]) / float(a[0] - a[3])
		coef2 = float(b[3] - b[2]) / float(a[3] - a[2])
		coef3 = float(b[2] - b[1]) / float(a[2] - a[1])
		coef4 = float(b[1] - b[0]) / float(a[1] - a[0])

	for x in range(min(a), min(max(a)+1,width)):
		for y in range(min(b), min(max(b)+1,height)):
			if y <= (coef1*x + b[0]-a[0]*coef1) and sign(coef2)*y >= sign(coef2)*(coef2*x + b[3]-a[3]*coef2) and y >= (coef3*x + b[2]-a[2]*coef3) and sign(coef4)*y <= sign(coef4)*(coef4*x + b[1]-a[1]*coef4) and img[height-y,x] != 255:
				count += 1
	return count
	