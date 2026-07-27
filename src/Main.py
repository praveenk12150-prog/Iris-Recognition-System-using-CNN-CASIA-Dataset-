from tkinter import *
from tkinter import filedialog, messagebox
import numpy as np
import os
import cv2
from keras.models import Sequential, model_from_json
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import pickle

main = Tk()
main.title("Iris Recognition using Machine Learning Technique")
main.geometry("1300x700")

global model, text
model = None
count = 0
miss = []

def uploadDataset():
    global datasetpath
    datasetpath = filedialog.askdirectory(initialdir=".")
    text.insert(END, f"{datasetpath} loaded")
def loadModel():
    global model
    text.delete('1.0', END)

    if not os.path.exists('model/X.txt.npy') or not os.path.exists('model/Y.txt.npy'):
        text.insert(END, "Error: Training data files 'X.txt.npy' or 'Y.txt.npy' not found in 'model' folder.\n")
        return

    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    text.insert(END, f'Dataset contains total {X_train.shape[0]} iris images from {Y_train.shape[1]}\n')

    if os.path.exists('model/model.json') and os.path.exists('model/model_weights.h5'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()
        print(model.summary())

        if os.path.exists('model/history.pckl'):
            with open('model/history.pckl', 'rb') as f:
                data = pickle.load(f)
            acc = data.get('accuracy', [])
            if acc:
                accuracy = acc[-1] * 100
                text.insert(END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
        else:
            text.insert(END, "Warning: 'history.pckl' not found. Accuracy graph may not be available.\n\n")

        text.insert(END, "See Black Console to view CNN layers\n")

    else:
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim=256, activation='relu'))
        model.add(Dense(output_dim=108, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=60, shuffle=True, verbose=2)
        model.save_weights('model/model_weights.h5')
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        with open('model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
        acc = hist.history.get('accuracy', [])
        if acc:
            accuracy = acc[-1] * 100
            text.insert(END, f"CNN Model Prediction Accuracy = {accuracy:.2f}%\n\n")
        text.insert(END, "See Black Console to view CNN layers\n")

def graph():
    if not os.path.exists('model/history.pckl'):
        messagebox.showerror("Error", "Model history not found. Train the model first.")
        return

    with open('model/history.pckl', 'rb') as f:
        data = pickle.load(f)

    accuracy = data.get('accuracy', [])
    loss = data.get('loss', [])

    if not accuracy or not loss:
        messagebox.showerror("Error", "Model history data is empty.")
        return

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color='red')
    plt.plot(accuracy, 'ro-', color='green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('CNN Accuracy & Loss Graph')
    plt.show()

def getIrisFeatures(image):
    global count
    img = cv2.imread(image, 0)
    if img is None:
        print(f"Error loading image: {image}")
        messagebox.showerror("Error", f"Unable to load image: {image}")
        return None

    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=63, param2=70, minRadius=0, maxRadius=0)

    if circles is not None:
        height, width = img.shape
        mask = np.zeros((height, width), np.uint8)
        for i in circles[0, :]:
            cv2.circle(mask, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), thickness=0)
            masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
            _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0][0])
            crop = img[y:y + h, x:x + w]
            cv2.imwrite("test.png", crop)
    else:
        count += 1
        miss.append(image)
        messagebox.showwarning("Warning", "No iris circle detected in image.")

    if os.path.exists("test.png"):
        return cv2.imread("test.png")
    else:
        return None

def predictChange():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    test_image = getIrisFeatures(filename)
    if test_image is None:
        return

    img = cv2.resize(test_image, (64, 64))
    im2arr = np.array(img).reshape(1, 64, 64, 3).astype('float32') / 255
    preds = model.predict(im2arr)
    predict = np.argmax(preds) + 1

    print("Predicted Class:", predict)

    original_img = cv2.imread(filename)
    if original_img is not None:
        original_img = cv2.resize(original_img, (600, 400))
        cv2.putText(original_img, f'Person ID Predicted from Iris Recognition is : {predict}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Prediction', original_img)

    if os.path.exists("test.png"):
        iris_img = cv2.imread("test.png")
        iris_img = cv2.resize(iris_img, (400, 200))
        cv2.imshow('Iris features extracted from image', iris_img)

    cv2.waitKey(0)

font = ('times', 14, 'bold')
title = Label(main, text='Iris Recognition using Machine Learning Technique', justify=LEFT)
title.config(bg='orange', fg='black')
title.config(font=font)
title.place(x=0, y=5, width=1500, height=40)

text = Text(main, height=25, width=130)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=60)

uploadButton = Button(main, text="Upload Iris Dataset", command=uploadDataset)
uploadButton.place(x=10, y=650)

modelButton = Button(main, text="Generate & Load CNN Model", command=loadModel)
modelButton.place(x=180, y=650)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=400, y=650)

predictButton = Button(main, text="Upload Iris Test Image & Recognize", command=predictChange)
predictButton.place(x=600, y=650)

exitButton = Button(main, text="Exit", command=main.destroy)
exitButton.place(x=880, y=650)

main.mainloop()
