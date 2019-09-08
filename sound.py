# importing the pyttsx library 
import pyttsx3 
  
# initialisation 
engine = pyttsx3.init() 
  
# testing 
engine.say("This is an example") 
engine.say("say the label") 
engine.runAndWait() 
