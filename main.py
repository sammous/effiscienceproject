#-*- coding:utf-8 -*-

import horseTracking as hT
import displayAndGraphs as dG
import processings as pr
import pylab as p
import time


def main(video,nb_image,process_ratio,taille,ombre,display,direction,process_ratio_b):
    r = pr.Resultat(video,nb_image,process_ratio)
    r_Tracking = hT.main(r,display)
    r_Processed = pr.main(r_Tracking,display)
    dG.main(r_Processed,taille,ombre,direction,process_ratio_b)
    return r_Processed



if __name__ == '__main__':
    debut=time.time()
    #status = main(video="video/base9.avi",nb_image=3,process_ratio=2,taille=160,ombre=False,display=False,direction='GD',process_ratio_b=2)
    #status = main(video="video/base11.avi",nb_image=3,process_ratio=2,taille=88,ombre=False,display=False,direction='DG',process_ratio_b=3)
    #status = main(video="video/base4.avi",nb_image=3,process_ratio=2,taille=120,ombre=True,display=False,direction='GD',process_ratio_b=2)
    status = main(video="video/base5.avi",nb_image=3,process_ratio=2,taille=100,ombre=True,display=True,direction='DG',process_ratio_b=3)

    #status = main(video="video/base7.avi",nb_image=3,process_ratio=2,taille=135,ombre=True,display=False,direction='DG',process_ratio_b=3)
    #status = main(video="video/base.avi",nb_image=3,process_ratio=2,taille=65,ombre=False,display=False,direction='GD',process_ratio_b=2)
    #status = main(video="video/base1.avi",nb_image=3,process_ratio=2,taille=65,ombre=False,display=False,direction='GD',process_ratio_b=5)

    fin=time.time()
    print(fin-debut)
