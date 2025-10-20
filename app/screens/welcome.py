import os, sys

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton

from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ObjectProperty

# get path details
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
    favicon = os.path.join(base_path, "data/images/favicon.png")
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
    favicon = os.path.abspath(os.path.join(base_path, "..", "data/images/favicon.png"))

Builder.load_string('''

<WelcomeScreen>:
    #on_enter: app.update_chatbot_welcome(self)

    MDBoxLayout: # main box
        orientation: 'vertical'
        padding: dp(16)
        spacing: dp(24)

        Image:
            source: root.fav_path
            fit_mode: 'contain'
            size_hint_y: 0.4

        MDLabel:
            font_style: "H6"
            halign: 'center'
            adaptive_height: True
            markup: True
            text: "Your Private & Offline AI-Chatbot from [b][color=#2196F3]DasLearning.in[/color][/b]"

        MDLabel:
            font_style: "Body1"
            halign: 'center'
            adaptive_height: True
            text: "You will have a drop-down to select model and another drop-down to select maximum word limit for AI answer"

        MDLabel:
            id: download_stat
            font_style: "Subtitle1"
            halign: 'center'
            adaptive_height: True
            text: "Click Start to go to the chatbot screen"

        MDFillRoundFlatButton:
            id: btn_start_chat
            pos_hint: {'center_x': 0.5}
            size_hint_x: 0.4
            text: "Start"
            font_size: sp(18)
            on_release: app.start_from_welcome()

        Widget:
            size_hint_y: 0.1

''')

class WelcomeScreen(MDScreen):
    fav_path = StringProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'welcome_screen'
        self.fav_path = favicon
