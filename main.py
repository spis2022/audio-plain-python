

# May need to do pip3 install pyaudio numpy

from basic_demo import play_sound_demo
from sounds import *

if __name__=="__main__":
    #play_sound_demo()
    play_sound(duration=1.0, frequency=440.0, volume=0.5)

