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
from kivy.utils import platform

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
        padding: 8, root.top_pad, 8, root.bottom_pad
        spacing: dp(16)

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
            text: "You will have a drop-down to select a model and another drop-down to select maximum word limit for AI answer."

        MDLabel:
            font_style: "Body1"
            halign: 'center'
            adaptive_height: True
            markup: True
            text: "You can select your local [i][b]PDF[/b][/i] or [i][b]DOCX[/b][/i] file to ask questions on the document. [b]On Android,[/b] you need to change the file extention from [i][b][color=#f58e2f].pdf[/color][/b][/i] to [i][b][color=#1c5c23].pdf.jpg[/color][/b][/i] & same for [i][b][color=#f58e2f].docx[/color][/b][/i] to [i][b][color=#1c5c23].docx.jpg[/color][/b][/i] before select from the app (due to permission issues)."

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
    top_pad = NumericProperty(0)
    bottom_pad = NumericProperty(0)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'welcome_screen'
        self.fav_path = favicon
        if platform == "android":
            try:
                from android.display_cutout import get_height_of_bar
                self.top_pad = int(get_height_of_bar('status'))
                self.bottom_pad = int(get_height_of_bar('navigation'))
            except Exception as e:
                print(f"Failed android 15 padding: {e}")
                self.top_pad = 32
                self.bottom_pad = 48
