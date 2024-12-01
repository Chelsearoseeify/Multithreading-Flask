# new_producer.py
import socket
import random
import time
import cv2
import numpy as np
import pytesseract
import threading
import re
import sys
import json


pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


# Cattura video dalla videocamera
cap = cv2.VideoCapture(1)

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


host = '127.0.0.1'
port = 12345
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


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

def init_draw_rectangle():
    global flag, result, current_rect, rects, key
    cv2.setMouseCallback("pd", draw_rectangle)
    print("selezione rettangoli")
    while flag==2:
        #devo fornire le coordinate che voglio che i punti assumano alla fine della correzione, sostanzialmente posso mettere la dimensione dello schermo
        cv2.imshow('pd', result)
        key = cv2.waitKey(1)
        if key == 13 and len(rects) == 2:  # Invio per uscire
            cv2.destroyAllWindows()
            flag = 3

        if current_rect:
            result = cv2.rectangle(result, (current_rect[0], current_rect[1]), (current_rect[2], current_rect[3]), (0, 255, 0), 2)

        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Funzione per applicare una maschera binaria all'interno dei rettangoli
def apply_binary_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #gray=cv2.GaussianBlur(gray,(5,5),0)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary=cv2.erode(binary, kernel, iterations=1)
    return binary

def run_timer(conn):
    timer_interrupt(conn)
    if flag != 4:  # As long as flag is not set to 4, keep repeating
        threading.Timer(1, run_timer, args=[conn]).start()


def timer_interrupt(conn):
    global numbers,t,minute,n1,n2,index, index_mean,flag,phase,index_post
    #recognize_numbers(result)
    #print("Timer interrupt eseguito")
    if flag==3:
      #print('siamo nell IF')
      n1 = [int(num) for num in re.findall(r'\((\d+)\)', numbers_old[0])]
      n1 = [random.randint(1, 100), random.randint(1, 100), 6]

      n2= [int(num) for num in re.findall(r'\((\d+)\)', numbers_old[1])]
      print('n1= ',n1)
      print('n2= ',n2)
      
      json_data = json.dumps(n1)  # Serialize the list to a JSON string
      conn.sendall(json_data.encode('utf-8'))  # Send the JSON string as bytes
      print(f"Producer: Sent {n1}")
      
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
        close_connection(conn)
        return



def reading_from_frames():
    global cap, key, start_interrupt, result, rects, coordinate_schermo, server_socket, host, port 
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
        # Set up a repeating timer with conn passed as an argument
        start_interrupt=0
        wait_for_connection(host, port, server_socket)
        conn = accept_connectionn(server_socket)
        timer = threading.Timer(1, run_timer, args=[conn])
        timer.start()
        
    # cv2.imshow("Result", result)
    cv2.imshow('Trasformazione', result)
    key = cv2.waitKey(1)

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

def capture_frame():
    global flag
    ret, frame = cap.read()
    if not ret:
        print("Errore nella cattura del frame")

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 13:  # Invio
        flag = 1
        cv2.imwrite('frame.jpg', frame)
        cv2.destroyAllWindows()

def load_image():
    global flag, img, img_old, points, coordinate_schermo, key
    img = cv2.imread('frame.jpg')
    img_old=img
    cv2.imshow('frame', img)
    key = cv2.waitKey(1)
    points = []

    print("Clicca con il tasto sinistro sugli angoli dello schermo")

   

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

def warp_perspective():
    global coordinate_schermo, img, result, key
    pnts1 = np.float32(coordinate_schermo)
    pnts2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])

    matrix = cv2.getPerspectiveTransform(pnts1, pnts2)
    result = cv2.warpPerspective(img, matrix, (1280, 720))
    cv2.imshow('pd', result)
    key = cv2.waitKey(1)

# Funzione per selezionare i punti
def select_point(event, x, y, flags, param):
    global k,img,img_old, points
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

def compute_index_status(index_pre, index_post):
    if index_pre and index_post:
        target = index_post / index_pre
        if target < 0.9:
            return ":)"
        return ":("
    return None

def clip_placed():
    global phase, index_mean, index, index_pre, minute, t
    print('clip posizionata')
    t=0
    minute=0
    index_pre=np.median(index_mean)
    phase=2
    index_mean=[]
    index=[]

def generate_data(conn):
    # Generate random data
    number = random.randint(1, 100)
    conn.sendall(str(number).encode('utf-8'))
    print(f"Producer: Sent {number}")
    time.sleep(1)  # Wait 1 second before sending the next number

def open_connection():
    host = '127.0.0.1'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Producer: Waiting for connection...")

    conn, addr = server_socket.accept()
    print(f"Producer: Connected to {addr}")

    try:
        while True:
            generate_data(conn)   
    except KeyboardInterrupt:
        print("Producer: Shutting down.")
    finally:
        conn.close()
        server_socket.close()


def wait_for_connection(host, port, server_socket):
    server_socket.bind((host, port))
    server_socket.listen(1)
    print("Producer: Waiting for connection...")

def accept_connectionn(server_socket):
    conn, addr = server_socket.accept()
    print(f"Producer: Connected to {addr}")
    return conn

def close_connection(conn, server_socket):
    conn.close()
    server_socket.close()


def main():
    global cap, flag, result, key, start_interrupt, phase, index_post, rects, coordinate_schermo, t, minute, index_mean, index, index_pre
    if not cap.isOpened():
        print("Non riesco ad aprire la videocamera")
        exit()
    else:
        print("Premi Invio quando la telecamera è ferma")


    # Flag per il controllo dello stato

    # Cattura un frame fino a quando non premi Invio
    while flag == 0:
        capture_frame()

    # Carica l'immagine salvata
    if flag == 1:
        load_image()

    while phase<3 and flag<4:
    #correzione di prospettiva
     if flag==2:
        warp_perspective()
        init_draw_rectangle()

     if flag==3:
        while flag==3:
            reading_from_frames()

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
                clip_placed()


            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                flag=4
                break


    if phase==3:
        compute_index_status(index_pre, index_post)
       





if __name__ == "__main__":
    main()
