#############################################################################
#                                                                           #
#      RECONNAISSANCE DE CHATS & CHIENS PAR TRANSFERT LEARNING => Demo      #
#       (dataset : https://www.kaggle.com/chetankv/dogs-cats-images)        #
#                                                                           #
#############################################################################

# Bibliothèque PIL pour utiliser les modules Image et ImageTk
from PIL import Image,ImageTk
# Bibliothèque tkinter
import tkinter as tk                # module de base
import tkinter.filedialog as tk_fd  # navigateur de fichiers
# Bibliothèques tensorflow
import tensorflow as tf

# Chargement du réseau DeepLearning à utiliser
C_TailleInput = (299,299)
MonReseau = tf.keras.models.load_model('MonReseau.h5')

# Création de la fenêtre principale (Window Root)
W_Root=tk.Tk()
W_Root.title('Reconnaissance de photos "chat ou chien"  @ JC Riat')
W_Root.geometry("+10+10")           # Position en haut à gauche
W_Root.resizable(False,False)       # Non redimensionnable
W_Root.configure(bg="darkgrey")     # Fond gris de la fenêtre

# Chargement des images utilisées dans le programme
C_inWidth, C_inHeight = (500,500)
ImgPIL = Image.new(mode="RGB", size=(C_inWidth, C_inHeight), color="linen")
ImgInputDef = ImageTk.PhotoImage(ImgPIL)
C_outWidth, C_outHeight = (200,200)
ImgPIL = Image.new(mode="RGB", size=(C_outWidth, C_outHeight), color="whitesmoke")
ImgOutputDef = ImageTk.PhotoImage(ImgPIL)
ImgPIL = Image.new(mode="RGB", size=(C_outWidth, C_outHeight), color="mistyrose")
ImgOutputCat = ImageTk.PhotoImage(ImgPIL)
ImgPIL = Image.new(mode="RGB", size=(C_outWidth, C_outHeight), color="azure")
ImgOutputDog = ImageTk.PhotoImage(ImgPIL) 

# Label pour afficher l'image à reconnaître
L_Input = tk.Label(W_Root, 
                   compound='center',                 # Label avec image+texte
                   image=ImgInputDef,                 # Image par défaut
                   width=C_inWidth,height=C_inHeight, # Taille fixe pour toutes les images     
                   font=('Arial',24,'bold'),          # Police de caractère
                   text='Image à identifier',         # Texte affiché sous l'image
                   bg='white',bd=3,relief='ridge')    # Mise en forme du label
L_Input.image = ImgInputDef                           # Pour stocker l'image avec l'objet
L_Input.pack(side=tk.LEFT, padx=10, pady=10)          # Affichage à gauche de la fenêtre

#------------------------------------------------
# Action du bouton B_Select
def Fct_Select():
  # Sélection du chemin avec le nom du fichier
  Chemin=tk_fd.askopenfilename(initialdir='..',
                               title="Choix de l'image à analyser",
                               filetypes=[('','.jfif'),('','.jpeg'),('','.jpg'),('','.png')])
  # Chargement de l'image en mémoire (conversion RGB nécecessaire si image en N&B)
  ImgPIL=Image.open(Chemin).convert('RGB')
  # Redimensionnement pour tenir dans la fenêtre C_inWidth x C_inHeight
  if ImgPIL.width > ImgPIL.height:
    ImgPIL=ImgPIL.resize((C_inWidth,round(ImgPIL.height/ImgPIL.width*C_inWidth)))
  else:
    ImgPIL=ImgPIL.resize((round(ImgPIL.width/ImgPIL.height*C_inHeight),C_inHeight))
  # Mise au format pour l'affichage dans le label
  ImgInput=ImageTk.PhotoImage(ImgPIL)
  # MàJ du label L_Input avec la nouvelle image à traiter
  L_Input["image"]=ImgInput
  L_Input["text"] =''
  L_Input.image=ImgInput
  # MàJ du label L_Output avec l'état par défaut 
  L_Output["image"]=ImgOutputDef
  L_Output.image=ImgOutputDef
  L_Output["text"]='Conclusion ?' 
#------------------------------------------------

# Bouton pour choisir l'image à reconnaître
B_Select = tk.Button(W_Root,
                     text="Sélectionner\nune image",  # Texte à afficher
                     font=('Arial',16),               # Police de caractère
                     command=Fct_Select)              # Fonction à appeler
B_Select.pack(padx=10,pady=10,expand=True,fill='x')   # Affichage à droite

#------------------------------------------------
# Action du bouton B_Lancer
def Fct_Lancer():
  # Chargement de l'image à analyser
  ImgPIL=ImageTk.getimage(L_Input.image).convert('RGB')
  # Image redimensionnée pour correspondre à la taille d'entrée du réseau
  ImgPIL=ImgPIL.resize(C_TailleInput) 
  # Transformation de l'image au format PIL en ndarray
  InputArray=tf.keras.utils.img_to_array(ImgPIL)/255
  # Création d'un lot avec une seule image (format en entrée du réseau)
  InputBatch=tf.expand_dims(InputArray, axis=0)
  # Calcul de la conclusion --> réel entre 0 (chat) et 1 (chien)
  Conclusion=MonReseau(InputBatch,training=False)[0][0]
  # MàJ du label L_Output avec la conclusion
  if Conclusion < 0.5:
    Confiance = 1-2*Conclusion
    L_Output["image"]=ImgOutputCat
    L_Output.image=ImgOutputCat
    L_Output["text"]='CHAT\n\nConfiance {:.2f} %'.format(100*Confiance)
  else:
    Confiance = 2*Conclusion-1
    L_Output["image"]=ImgOutputDog
    L_Output.image=ImgOutputDog
    L_Output["text"]='CHIEN\n\nConfiance {:.2f} %'.format(100*Confiance)
#------------------------------------------------

# Bouton pour lancer la reconnaissance
B_Lancer = tk.Button(W_Root,
                     text="Lancer la\nreconnaissance",# Police de caractère
                     font=('Arial',16),               # Police de caractère
                     command=Fct_Lancer)              # Fonction à appeler
B_Lancer.pack(padx=10,pady=10,expand=True,fill='x')   # Affichage à droite

# Label pour afficher le résultat de la reconnaissance
L_Output = tk.Label(W_Root,
                    compound='center',                # Label avec image+texte
                    image=ImgOutputDef,               # Image par défaut
                    font=('Arial',16),                # Police de caractère
                    text='Conclusion ?',              # Texte affiché sous l'image
                    bg='white',bd=3,relief='ridge')   # Mise en forme du label
L_Output.pack(padx=10,pady=10,expand=True)            # Affichage à droite

# Bouton pour sortir de l'application
B_Sortir = tk.Button(W_Root,
                     text='SORTIR',                   # Texte à afficher
                     font=('Arial',16,'bold'),        # Police de caractère
                     command=W_Root.destroy)          # Fonction à appeler
B_Sortir.pack(ipady=10,padx=10,pady=10,expand=True,fill='x')

# Gestion de la fenêtre principale
W_Root.mainloop()