import pyglet
import time
import os


def speak(obj, pos):
    obj = obj.replace(' ', '_')
    file1 = os.path.join(r"F:\Datasets\Audio Files\Objects", str(obj) + ".mp3")
    file2 = os.path.join(r"F:\Datasets\Audio Files\Position", str(pos) + ".mp3")
    music = pyglet.media.load(file1, streaming=False)
    # speaking.clear()
    music.play()
    time.sleep(music.duration)
    music = pyglet.media.load(file2, streaming=False)
    # speaking.clear()
    music.play()
    time.sleep(music.duration)
    # speaking.set()


# speak('cell phone', 'front')

