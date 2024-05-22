from tkinter import *
import tkinter.font as font
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from final_gui import detect_and_recognize_people,add_user

Top,imgpath,choice=0,0,-1
user_name,nm,clicked='','',''

def goodbye():
    messagebox.showinfo("Observer","SEE YOU SOON")
    root.destroy()
    
def destroyTop1():
    global Top,user_name,nm,imgpath
    user_name=nm.get()
    picpath=Image.open(imgpath)
    user_name.replace(' ','_')
    picpath.save("F:\\Coding\\Python_VS_Code\\sec_proj\\photos\\{}.jpg".format(user_name))
    add_user(user_name)
    Top.destroy()
    Top.update()
    
def destroyTop2():
    global Top
    Top.destroy()
    Top.update()

def config(root):
    root.title("Observer")
    root.geometry("1350x700+0+0")
    root.state("zoom")
    root.config(bg='black')
    
def open_image():
    global imgpath,Top
    imgpath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp *.ppm *.pgm")])
    if imgpath:
        lbpath=Label(Top,text=imgpath)
        lbpath.place(x=80,y=200)

'''
def get_cam_choice(choice):
    global clicked
    choice=clicked.get()
    detect_and_recognize_people(choice)
    
def camchoice():
    global clicked
    options=[0,1,2,3,4]
    clicked=StringVar()
    clicked.set(0)
    drop=OptionMenu(root, clicked, *options,command=get_cam_choice)
    drop.configure(fg='white',bg='black',font=('Arial 20'))
    drop.place(x=1150,y=720)

'''

def reg():
    global Top,nm
    # stp.destroy()
    # stp.update()
    Top=Toplevel()
    Top.geometry('600x300')
    Top.title('USER REGISTRATION')
    #Top.configure(bg='#333333')
    img=PhotoImage(file="F:\\Coding\\Python_VS_Code\\sec_proj\\myimage.png")
    
    backg=Label(Top,image=img)
    backg.pack()
    backg.image=img
    lb1 = Label(Top, text='NAME:', font=('arial', 20, 'bold'))
    lb1.place(x=20,y=20)
    nm = Entry(Top, font=('arial', 15, 'bold'))
    nm.place(x=150,y=25)
    lb2 = Label(Top, text='BROWSE YOUR PHOTO:', font=('arial', 20, 'bold'))
    lb2.place(x=140,y=90)
    myfont=font.Font(size=15)
    open_button = Button(Top, text="Open Image", command=open_image)
    open_button.place(x=245,y=140)
    open_button['font']=myfont
    
    myfont2=font.Font(size=15)
    submit = Button(Top, text="Submit",command=destroyTop1)
    submit.place(x=60,y=250)
    submit['font']=myfont2
    cancel = Button(Top, text="Cancel",command=destroyTop2)
    cancel.place(x=450,y=250)
    cancel['font']=myfont2

def step2():
    ps = passwrd.get()
    ps = ps.upper()
    if ps == 'ALPHA007':
        lb1 = Label(stp, text='HELLO SIR, CLICK CONTINUE TO PROCEED', font=('times new roman', 10, 'bold'))
        lb1.place(x=10, y=90)
        cont = Button(stp, text='CONTINUE', command=reg)
        cont.place(x=10, y=120)
    else:
        lb2 = Label(stp, text='ACCESS DENIED', font=('times new roman', 10, 'bold'))
        lb2.place(x=10, y=100)

def step1():
    global stp
    stp = Toplevel()
    stp.geometry('350x150')
    stp.title('PASS')
    stp.configure(bg='#000000')
    ps = Label(stp, text='ENTER PASSWORD:', font=('times new roman', 13, 'bold'))
    ps.place(x=10, y=10)
    global passwrd
    passwrd = Entry(stp, show='*', font=('times new roman', 15, 'bold'))
    passwrd.place(x=10, y=50)
    accept = Button(stp, text='->', command=step2)
    accept.place(x=250, y=50)

def animation(count):
    global showAnination
    newImage = imageObject[count]

    gif_Label.configure(image=newImage)
    count += 1
    if count == frames:
        count = 0
    
    showAnimation = root.after(50, lambda: animation(count))

def animation2(count2):
    global showAnination2
    newImage2 = imageObject2[count2]

    gif_Label2.configure(image=newImage2)
    count2 += 1
    if count2 == frames2:
        count2 = 0
    
    showAnimation2 = root.after(5, lambda: animation2(count2))

# *******************************************************************************************

root=Tk()

imgbg_pil = Image.open('F:/Coding/Python_VS_Code/sec_proj/jacketpotato.jpg')
imgbg = ImageTk.PhotoImage(imgbg_pil)
# imgbg=PhotoImage(file='F:/Coding/Python_VS_Code/sec_proj/city.png')
lb1=Label(root,image=imgbg)
lb1.pack()
config(root)
myfont=font.Font(size=30)
bn = Button(root, text='Add User',fg='silver',bg='DarkSlateGrey',command=reg)
bn['font']=myfont
bn.place(x=150,y=850)

# head1=Label(root, text='THE OBSERVER', font=('Bahnschrift Light', 50, 'bold'),fg='white',bg='grey51').pack()
# head2=Label(root, text="by the AlphaTrackers", font=('Bahnschrift',20,'bold'),fg='white',bg='black')
# head2.pack()

#******************************************************************************************************************

bn2 = Button(root,text="Start System",fg='silver',bg='DarkSlateGrey',command=detect_and_recognize_people)
bn2.place(x=800,y=850)
bn2['font']=myfont

bn3 = Button(root, text='Cancel', fg='silver',bg='DarkSlateGrey',command=goodbye)
bn3['font']=myfont
bn3.place(x=1650,y=850)

#******************************************************************************************************************

gifImage = r"F:\Coding\Python_VS_Code\sec_proj\mygif.gif"
openImage = Image.open(gifImage)
frames = openImage.n_frames
imageObject = [PhotoImage(file=gifImage, format=f"gif -index {i}") for i in range(frames)]
count = 0
showAnimation = None

gif_Label = Label(root, bg='black' ,image="")
gif_Label.place(x=430,y=400)
#*******************************************************************************************************************************
gifImage2 = r"F:\Coding\Python_VS_Code\sec_proj\mygif3.gif"
openImage2 = Image.open(gifImage2)
frames2 = openImage2.n_frames
imageObject2 = [PhotoImage(file=gifImage2, format=f"gif -index {i}") for i in range(frames2)]
count2 = 0
showAnimation2 = None

gif_Label2 = Label(root, bg='black' ,image="")
gif_Label2. place(x=380,y=0)

animation2(count2)

#********************************************************************************************************************************
animation(count)
root.mainloop()