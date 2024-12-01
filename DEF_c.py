import cv2
import numpy as np
import pytesseract
import threading
import re
import sys


pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


# Cattura video dalla videocamera
cap = cv2.VideoCapture(0)

# Variabili globali
coordinate_schermo=[]
numbers = []
numbers_old=[]
img=[]
img_old=[]
k=0
rects = []  # Lista per memorizzare i rettangoli selezionati
trackers = []  # Lista per memorizzare i tracker
current_rect = None  # Rettangolo attualmente in fase di disegno
drawing = False  # True se il mouse è premuto
ix, iy = -1, -1  # Posizione iniziale del rettangolo
rettangoli=0
t=0
minute=0
n1=[]
n2=[]
index=[]
index_mean=[]
index_pre=[]
index_post=[]
phase=1
flag = 0
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
start_interrupt=1


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, current_rect, rects, trackers

    # Gestisci fino a due rettangoli
    if len(rects) < 2:
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing and len(rects) < 2:
                drawing = True
                ix, iy = x, y
                current_rect = (ix, iy, x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                current_rect = (ix, iy, x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if drawing :
                drawing = False
                rect = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
                rects.append(rect)
                current_rect = None

    if event == cv2.EVENT_RBUTTONDOWN:
        rects.pop(-1)
                # Crea e aggiungi un tracker per il rettangolo appena aggiunto
                #tracker = cv2.TrackerCSRT_create()
                #tracker.init(frame, tuple(rect))
                #trackers.append(tracker)


# Funzione per applicare una maschera binaria all'interno dei rettangoli
def apply_binary_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray=cv2.GaussianBlur(gray,(5,5),0)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary=cv2.erode(binary, kernel, iterations=1)
    return binary

def timer_interrupt():
    global numbers,t,minute,n1,n2,index, index_mean,flag,phase,index_post
    #recognize_numbers(result)
    #print("Timer interrupt eseguito")
    if flag==3:
      #print('siamo nell IF')
      n1 = [int(num) for num in re.findall(r'\((\d+)\)', numbers_old[0])]
      n2= [int(num) for num in re.findall(r'\((\d+)\)', numbers_old[1])]
      print('n1= ',n1)
      print('n2= ',n2)
      if len(n1)>0 and len(n2)>0:
          if n1[0]>0 and n2[0]>0:
            index.append(n2[0]/n1[0])

      t=t+1
      if (t==10):
        t=0
        index_mean.append(np.median(index))
        index=[]
        minute=minute+1
        if minute>=6:
            if phase==1:
             index_mean.pop(0)
            if phase==2:
              index_post=np.median(index_mean)
              phase=3
              flag=4
    if flag==4:
        return

    threading.Timer(1, timer_interrupt).start()  # Intervallo di 1 secondo








# Funzione per riconoscere e stampare i numeri dai rettangoli
def recognize_numbers(result):
    global numbers, numbers_old
    for rect in rects:
        x, y, w, h = rect
        roi = result[y:y + h, x:x + w]
        binary_roi = apply_binary_mask(roi)
        cv2.imshow('binario',binary_roi)
        number = pytesseract.image_to_string(binary_roi,
                                             config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/?()')
        numbers.append(number.strip())
    if len(numbers) == 2:
        #print(f"Numero rettangolo 1: {numbers[0]}; Numero rettangolo 2: {numbers[1]}")
        numbers_old=numbers
        numbers=[]


if not cap.isOpened():
    print("Non riesco ad aprire la videocamera")
    exit()
else:
    print("Premi Invio quando la telecamera è ferma")


  # Flag per il controllo dello stato

# Cattura un frame fino a quando non premi Invio
while flag == 0:
    ret, frame = cap.read()
    if not ret:
        print("Errore nella cattura del frame")
        break

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 13:  # Invio
        flag = 1
        cv2.imwrite('frame.jpg', frame)
        cv2.destroyAllWindows()

# Carica l'immagine salvata
if flag == 1:

    img = cv2.imread('frame.jpg')
    img_old=img
    cv2.imshow('frame', img)
    key = cv2.waitKey(1)
    points = []




    print("Clicca con il tasto sinistro sugli angoli dello schermo")
    # Funzione per selezionare i punti
    def select_point(event, x, y, flags, param):
        global k,img,img_old
        if event == cv2.EVENT_LBUTTONDOWN:  # Se si fa clic con il tasto sinistro del mouse
            points.append((x, y))  # Aggiungi le coordinate alla lista
            print(f"Punto selezionato: ({x}, {y})")  # Stampa le coordinate
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Disegna un cerchio rosso


            # Se hai bisogno di selezionare solo 4 punti, puoi aggiungere un controllo
            if len(points) >= 4:
                print("Hai selezionato 4 punti. Premi Invio per uscire.")

            if len(points) == k+2:
                cv2.circle(img_old, points[k], 5, (0, 0, 255), -1)
                k=k+1

        if event == cv2.EVENT_RBUTTONDOWN:
            points.pop()
            img=img_old
            print('punti:' ,points)







    cv2.setMouseCallback('frame', select_point)
    coordinate_schermo = [(0, 0)] * 4

    # Ciclo per mantenere la finestra aperta
    while True:
        cv2.imshow('frame', img)
        key = cv2.waitKey(1)
        if key == 13:  # Invio per uscire
            cv2.destroyAllWindows()
            flag=2
            coordinate_schermo[0]= min(points, key=lambda point: (point[1]+point[0]))
            coordinate_schermo[3]= max(points, key=lambda point: (point[1]+point[0]))
            coordinate_schermo[2]=min(points, key=lambda point: (-point[1]+point[0]))
            coordinate_schermo[1]=min(points, key=lambda point: (point[1]-point[0]))

            break


while phase<3 and flag<4:
#correzione di prospettiva
 if flag==2:

    pnts1 = np.float32(coordinate_schermo)
    pnts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])

    matrix = cv2.getPerspectiveTransform(pnts1, pnts2)
    result = cv2.warpPerspective(img, matrix, (1280, 720))
    cv2.imshow('pd', result)
    key = cv2.waitKey(1)



    cv2.setMouseCallback("pd", draw_rectangle)
    print("selezione rettangoli")
    while flag==2:
        #_, video = cap.read()
        #stampo l'immagine con i cerchi (si può anche togliere)
        #cv2.circle(video, coordinate_schermo[0], 5, (0, 0, 255), -1 )
        #cv2.circle(video, coordinate_schermo[1], 5, (0, 0, 255), -1)
        #cv2.circle(video, coordinate_schermo[2], 5, (0, 0, 255), -1)
        #cv2.circle(video, coordinate_schermo[3], 5, (0, 0, 255), -1)


        #cv2.imshow('frame', video)
        #key=cv2.waitKey(1)

        #devo fornire le coordinate che voglio che i punti assumano alla fine della correzione, sostanzialmente posso mettere la dimensione dello schermo

        cv2.imshow('pd', result)
        key = cv2.waitKey(1)
        if key == 13 and len(rects) == 2:  # Invio per uscire
            cv2.destroyAllWindows()
            flag = 3


        #if rettangoli==0:
        #cv2.setMouseCallback("Trasformazione", draw_rectangle)
        #rettangoli=1

        #for i, tracker in enumerate(trackers):
        #    success, rect = tracker.update(result)
        #    if success:
        #        rects[i] = tuple(map(int, rect))  # Aggiorna il rettangolo con la nuova posizione

        if current_rect:
            result = cv2.rectangle(result, (current_rect[0], current_rect[1]), (current_rect[2], current_rect[3]), (0, 255, 0), 2)

        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)




 if flag==3:
    while flag==3:
        _, video = cap.read()
        pnts1 = np.float32(coordinate_schermo)
        pnts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])

        matrix = cv2.getPerspectiveTransform(pnts1, pnts2)
        result = cv2.warpPerspective(video, matrix, (1280, 720))

        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        recognize_numbers(result)

        if start_interrupt==1:
            timer_interrupt()
            start_interrupt=0

        # cv2.imshow("Result", result)
        cv2.imshow('Trasformazione', result)
        key = cv2.waitKey(1)

        if key == 8:  #backspace
            rects = []
            flag=2
            t = 0
            minute = 0
            index_mean = []
            index = []
            cv2.destroyAllWindows()

            #trackers = []  # Resetta anche i tracker

        if (key == 13 and phase==1):
            print('clip posizionata')
            t=0
            minute=0
            index_pre=np.median(index_mean)
            phase=2
            index_mean=[]
            index=[]


        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            flag=4
            break


if phase==3:
    target=index_post/index_pre
    print('index pre:',index_pre,'\nindex post:',index_post)
    if target<0.9:
        print(':)')
    else:
        print(':(')










