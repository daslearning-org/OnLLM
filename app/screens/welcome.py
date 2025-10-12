from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton

from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.metrics import dp, sp

Builder.load_string('''

<WelcomeScreen>:
    #on_enter: app.update_chatbot_welcome(self)

    MDBoxLayout: # main box
        orientation: 'vertical'
        padding: dp(8)
        spacing: dp(10)

        Image:
            source: 'data/images/favicon.png'
            fit_mode: 'contain'
            size_hint_y: 0.2

        MDLabel:
            font_style: "H5"
            halign: 'center'
            adaptive_height: True
            text: "Your Private & Offline AI-Chatbot"

        MDLabel:
            id: download_stat
            font_style: "Subtitle1"
            halign: 'center'
            adaptive_height: True
            text: "Click start button to check if the model files are present"

        MDFillRoundFlatButton:
            id: btn_start_chat
            pos_hint: {'center_x': 0.5}
            size_hint_x: 0.4
            text: "Start"
            on_release: app.start_from_welcome()

''')

class WelcomeScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'welcome_screen'
