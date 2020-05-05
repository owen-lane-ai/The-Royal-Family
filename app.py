import tkinter as tk
from tkinter import filedialog, Text
import os

root = tk.Tk()
apps = []

if os.path.isfile('save.txt'):
    with open('save.txt', 'r') as f:
        tempApps = f.read()
        tempApps = tempApps.split(',')
        apps = [x for x in tempApps if x.strip()]

def addApp():

    for widget in frame.winfo_children():
        widget.destroy()

    filename= filedialog.askopenfilename(initialdir="/", title="Select File", 
                                        filetypes=(("executables","*.exe"), ("all files", "*.*")))
    apps.append(filename)
    print(filename)
    for app in apps:
        label = tk.Label(frame, text=app, bg="gray")
        label.pack()

def runApps():
    for app in apps:
        os.startfile(app)




canvas = tk.Canvas(root, height=500, width=1000, bg="#263D42")
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.6, relheight=0.6, relx=0.05, rely=0.1)

frame2 = tk.Frame(root, bg="white")
frame2.place(relwidth=0.3, relheight=0.4, relx=0.69, rely=0.3)

openFile = tk.Button(root, text="Open File", padx=10, 
                    pady=5, fg="white", bg="#263D42", command=addApp)
openFile.pack()

runApps = tk.Button(root, text="Run Apps", padx=10, 
                    pady=5, fg="white", bg="#263D42", command=runApps)
runApps.pack()

exitButton = tk.Button(root, text="Exit", padx=10, 
                    pady=5, fg="white", bg="#263D42", command=root.destroy)
exitButton.pack()



for app in apps:
    label = tk.Label(frame, text=app)
    label.pack()


root.mainloop()

with open('save.txt', 'w') as f:
    for app in apps:
        f.write(app + ',')
