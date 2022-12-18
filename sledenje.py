import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

plt.ion()


class Sledilnik:
    def __init__(self, sledilno_okno, slika, direktorij_originali, direktorij_oznacene):
        self.slika = slika
        self.direktorij_originali = direktorij_originali
        self.direktorij_oznacene = direktorij_oznacene
        self.sledilno_okno = sledilno_okno
        self.kriterij = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

        x, y, w, h = self.sledilno_okno
        print("x: ", x, "y: ", y, "w: ", w, "h: ", h)
        print("slika: ", slika)

        self.sablona = self.slika[y:y + h, x:x + w] # to je sablona bahprej noramliziramo cv.normalize 
        cv.normalize( self.sablona,  self.sablona, 0, 255, cv.NORM_MINMAX)
        self.slika = cv.cvtColor(self.slika, cv.COLOR_BGR2GRAY)

        #potem pa spremenimo v sivo 
        #hsv_roi = cv.cvtColor(self.sablona, cv.COLOR_BGR2HSV)
        #mask = np.uint8(np.logical_and(hsv_roi[:, :, 1]>60, hsv_roi[:, :, 2]>32))
        #roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        #cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        #self.roi_hist = roi_hist
        #odštej povprečje

    def poisci_ujemanje(self):
        x, y, w, h = self.sledilno_okno
        jedro = np.ones((w, h))
        img_local_avg = cv.filter2D(src=self.slika, ddepth=-1, kernel=jedro) / (w*h)
        odsteta_slika = self.slika - img_local_avg

        flipd_sablona = np.flip(self.sablona)
        filtrirana_slika = cv.filter2D(src=odsteta_slika, ddepth=-1, kernel=flipd_sablona)

        kvadrirana_slika = np.square(self.sablona)
        sum_kvadrata = np.sum(kvadrirana_slika)

        konvulirana = cv.filter2D(src=odsteta_slika, ddepth=-1, kernel=jedro)

        R =  filtrirana_slika / np.sqrt(sum_kvadrata * konvulirana)
        return R




    def sledi(self, show_plot=False):
        for ime_slike in sorted(os.listdir(self.direktorij_originali)):
            if ime_slike.endswith(".jpg"):
                self.slika = cv.imread(os.path.join(self.direktorij_originali, ime_slike))
                rgb_slika = self.slika.copy()

                cv.normalize(self.slika, self.slika, 0, 255, cv.NORM_MINMAX) #normaliziramo
                self.slika = cv.cvtColor(self.slika, cv.COLOR_BGR2GRAY) #V SIVINSKO
                


                slika_verjetnosti = self.poisci_ujemanje()
                #slika_verjetnosti = cv.calcBackProject([self.slika], [0], self.roi_hist, [0, 180], 1)
                #mask = np.uint8(np.logical_and(self.slika[:, :, 1]>60, self.slika[:, :, 2]>32))
                #slika_verjetnosti[mask==False] = 0
                #prvo pretvorim v sivinsko 

                if(show_plot):
                    plt.figure(1)
                    plt.clf()
                    plt.imshow(slika_verjetnosti)
                    plt.colorbar()
                    plt.show()

                x, y, w, h = self.sledilno_okno
                w = w // 2
                h = h // 2
                x = x + (h // 2)
                y = y + (w // 2)
                majhno_kno = (x, y, w, h)
                ret, majhno_kno = cv.meanShift(slika_verjetnosti, majhno_kno, self.kriterij)
                self.sledilno_okno = (majhno_kno[0] - h // 2,
                                      majhno_kno[1] - w // 2,
                                      self.sledilno_okno[2],
                                      self.sledilno_okno[3])
                if(show_plot):  
                    plt.figure(2)
                    plt.clf()
                x, y, w, h = self.sledilno_okno
                img2 = cv.rectangle(rgb_slika, (x, y), (x + w, y + h), (255, 0, 255), 2)
                if(show_plot):  plt.imshow(img2)

                write_path = os.path.join(self.direktorij_oznacene, ime_slike)
                print("writing to path: ", write_path)
                cv.imwrite(write_path, img2) 
                if(show_plot):  plt.waitforbuttonpress(timeout=0.1)


    def naredi_posnetek(self, ime_videa):
        visina, sirina = self.slika.shape
        video = cv.VideoWriter(ime_videa, cv.VideoWriter_fourcc(*'MJPG'), 30, (sirina, visina))
        
        for ime_slike in sorted(os.listdir(self.direktorij_oznacene)):
            if ime_slike.endswith(".jpg"):
                self.slika = cv.imread(os.path.join(self.direktorij_oznacene, ime_slike))

                video.write(self.slika)
        
        video.release()