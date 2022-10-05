from cProfile import label
import os
os.environ["KIVY_VIDEO"] = "ffpyplayer"
#os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'
#os.environ['KIVY_CLIPBOARD'] = 'pygame'
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
import cv2
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.widget import Widget
import facelandmarks
import faceswapper
from kivy.graphics.texture import Texture
from kivy.animation import Animation
import ffmpeg


Window.size = (700,500)
Window.title = "Better"

class FirstScreen(Screen):
    pass

class MyWidget(BoxLayout):
    def selected(self, filesource):
        try:
            self.ids.my_image.source = filesource[0] 
        except:
            pass
    
    def getpath(self, filesource):
        return filesource[0]

    def facedetect(self, filesource):        
        try:
            #print(filesource)
            frame =  facelandmarks.facemarkdetectImage(filesource[0])
            buf1 = cv2.flip(frame, 0)
            buf =buf1.tostring()
            frame_texture = Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
            frame_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
            self.ids.my_image.texture = frame_texture
        except:
            print('Error!')

    def triangles(self, filesource):
        try:
            frame =  facelandmarks.Triangles(filesource[0])
            buf1 = cv2.flip(frame, 0)
            buf =buf1.tostring()
            frame_texture = Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
            frame_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
            self.ids.my_image.texture = frame_texture
        except:
            print('Error!')
    
    def getmask(self, filesource):
        try:
            frame =  facelandmarks.getMask(filesource[0])
            buf1 = cv2.flip(frame, 0)
            buf =buf1.tostring()
            frame_texture = Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
            frame_texture.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
            self.ids.my_image.texture = frame_texture
        except:
            print('Error!')


class MyWidget2(BoxLayout):
    pass

class ImageChoose(Screen):
    pass

class Vplay(BoxLayout):
    def playVideo(self, filesource):
        try:
            if (filesource[0].split('.')[-1] == 'mp4' or 'avi'):
                self.ids.video1.source = filesource[0]
            else:
                print('Error')
        except:
            print('Error') 

    def getpath(self, filesource):
        return filesource[0]


class VideoPlayScreen(Screen):
    pass

class Facer(Screen):
    def selected(self, filesource):
        try:
            self.ids.my_image.source = filesource[0] 
        except:
            pass

    def playVideo(self, filesource1):
        try:
            if (filesource1[0].split('.')[-1] == 'mp4' or 'avi'):
                self.ids.video1.source = filesource1[0]
            else:
                print('Error')
        except:
            print('Error') 
    
    def faceswap(self, filesource, filesource1):
        try:
            self.ids.video1.source = faceswapper.savevideo(filesource[0], filesource1[0])
        except:
            print("Error")

presentation = Builder.load_file("design.kv")

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(FirstScreen(name = 'main'))
        sm.add_widget(ImageChoose(name = 'imagechoose'))
        sm.add_widget(VideoPlayScreen(name = 'vplayscreen'))
        sm.add_widget(Facer(name = 'facer'))
        sm.current = 'main'
        return sm
    
if __name__ == '__main__':
    MyApp().run()
