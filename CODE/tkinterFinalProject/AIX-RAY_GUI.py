import tkinter as tk
from turtle import color
from PIL import Image, ImageTk
import cv2 
from tkinter import filedialog
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename
import os
import json

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('CODE/tkinterFinalProject/Sequential_model_1_224x224x3input_78.04acc.h5')


class TkinterApp:
    def __init__(self, window, window_title, image_path):

        self.BASEDIR = os.path.dirname(os.path.abspath(__file__))

        self.window = window
        self.window.title(window_title)

        PatientName = tk.Label(self.window, text="Patient  Name: ")
        PatientName.place(x=10, y=10)
        self.PatientNameEntry = tk.Entry(self.window, text="Patient Name: ")
        self.PatientNameEntry.place(x=150, y=10)


        PatientSurname = tk.Label(self.window, text="Patient Surname: ")
        PatientSurname.place(x=10, y=50)
        self.PatientSurnameEntry = tk.Entry(self.window, text="Patient Surname: ")
        self.PatientSurnameEntry.place(x=150, y=50)

       
        PatientID = tk.Label(self.window, text="Patient ID: ")
        PatientID.place(x=10, y=90)
        self.PatientIDEntry = tk.Entry(self.window, text="Patient ID: ")
        self.PatientIDEntry.place(x=150, y=90)


        # Checkbox for select the diagnosis name.
        diagnosis = ["Covid-19", "Pneumonia-VGG","Pneumonia-notVGG"]
        self.comboDiagnosis = ttk.Combobox(self.window, values=diagnosis)
        self.comboDiagnosis.set("Select Diagnosis")
        self.comboDiagnosis.place(x=522, y=10)


        searchImage = tk.Button(self.window, text="Select image",command=self.browseFile)
        searchImage.place(x=522, y=50)

        self.image_path_name_Label = tk.Label(self.window, text="Image name: ")
        self.image_path_name_Label.place(x=522, y=90)

        self.canvas = tk.Canvas(window, width=512, height=512)
        self.canvas.place(x=10, y=130)

        self.PILimage = Image.open(image_path)
        self.PILimage = self.PILimage.convert('RGB')
        image_resized = self.PILimage.resize((512, 512))
        self.image = ImageTk.PhotoImage(image_resized)
        self.image_container = self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)
     

        self.predictionButton = tk.Button(self.window, text="Predict", command=self.predict)
        self.predictionButton.place(x=10, y=700)


        predictionResult_text = tk.Label(self.window, text="Prediction Result: " )
        predictionResult_text.place(x=90, y =700)
        

        self.predictionResult_result = tk.Label(self.window, text="Please select image!")
        self.predictionResult_result.place(x=200, y=700)


        self.saveButton = tk.Button(self.window, text="Save Result", command=self.saveResultToTxt)
        self.saveButton.place(x=10, y=740)



        self.window.geometry("720x860")
        self.window.mainloop()

    def browseFile(self):
        # browse file and update canvas image.
        global filename
        filename = askopenfilename(
            filetypes=[("Image File", "*.jpg;*.jpeg;*.png;*.bmp")],
            title="Choose an image file.",
            )
        self.PILimage = Image.open(filename)
        self.PILimage = self.PILimage.convert('RGB')
        
        image_resized =  self.PILimage.resize((512, 512))
        self.image = ImageTk.PhotoImage(image_resized)
        self.canvas.itemconfig(self.image_container, image= self.image,anchor=tk.NW)


        image_name_splitted = filename.split("/")[-1]
        self.image_path_name_Label.config(text=image_name_splitted)

        print("filename", image_name_splitted)


    def predict(self):
        
        diagnosisName = self.comboDiagnosis.get()

        if diagnosisName == "Select Diagnosis":
            self.predictionResult_result.config(text="Please select diagnosis!",background="red")
            
        elif diagnosisName == "Covid-19": 
            result1 = self.zaturePrediction(modelNumber=0, inputSize=(224,224))

            if result1 > 0.5:
                self.predictionResult_result.config(text="Not Corona with probability of %{}".format(100*(result1[0])),background="green")

            else:   
                self.predictionResult_result.config(text="Corona with probability of %{}".format(100*(1-result1[0])),background="red")
            
        elif diagnosisName == "Pneumonia-VGG":
            # make prediction for zature.

            result = self.zaturePrediction(modelNumber=1, inputSize=(224,224))

            if result < 0.5:
                self.predictionResult_result.config(text="Not Pneumonia with probability of %{}".format(100*(1-result[0])),background="green")
    
            else:
                self.predictionResult_result.config(text="Pneumonia with probability of %{}".format(100*(result[0])),background="red")

        elif diagnosisName == "Pneumonia-notVGG":
            result2 = self.zaturePrediction(modelNumber=2,inputSize=(224,224))
            
            if result2 < 0.5:
                self.predictionResult_result.config(text="Not Pneumonia with probability of %{}".format(100*(1-result2[0])),background="green")

            else:
                self.predictionResult_result.config(text="Pneumonia with probability of %{}".format(100*(result2[0])),background="red")
                

            #self.predictionResult_result.config(text=str(result),background="green")

        else:
            self.predictionResult_result.config(text="Please select diagnosis!",background="red")
            

    
    def zaturePrediction(self, modelNumber = 0, inputSize = (224,224)):
        if modelNumber == 0:
            # load model 1.
            model = tf.keras.models.load_model('CODE/tkinterFinalProject/Covid_Model_VGG16_224x224x3input.h5')
        elif modelNumber == 1:
            # load model 2.
            model = tf.keras.models.load_model('CODE/tkinterFinalProject/TransferLearning224x224x3VGG16acc98.h5')
            
        elif modelNumber == 2:
            #load model 3
           model = tf.keras.models.load_model('CODE/tkinterFinalProject/Sequential_model_1_224x224x3input_78.04acc.h5')
        else:
            print("Please select valid model number!")

        image = self.PILimage.convert('RGB')

        # resize image to fit the model.
        image_resized = image.resize(inputSize)
        # convert image to numpy array.
        image_array = np.array(image_resized)
        image_array_expandDim = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array_expandDim / 255)

        return prediction


    def saveResultToTxt(self):
        
        print("save result")
        patientName = self.PatientNameEntry.get()
        patientSurname = self.PatientSurnameEntry.get()
        patientID = self.PatientIDEntry.get()
        diagnosisName = self.comboDiagnosis.get()
        image_name_splitted = filename.split("/")[-1]
        diagnosisResult = self.predictionResult_result.cget("text")

        # save result to json file as dictionary.
        with open('Patients/results.json', 'w') as f:
            json.dump({'patientName': patientName, 'patientSurname': patientSurname, 'patientID': patientID, 'diagnosisName': diagnosisName, 'image_name_splitted': image_name_splitted, 'diagnosisResult': diagnosisResult}, f)

        # read json file and print it.
        with open('Patients/results.json', 'r') as f:
            data = json.load(f)
            print(data)
    
if __name__ == "__main__":
    window = tk.Tk()
    TkinterApp(window, "AIXRAY", "CODE/AIX-RAY_logo.png")

