# screens/chatbot_screen.py
import sys, os

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton, MDFillRoundFlatIconButton

from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ObjectProperty

# get path details
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
noto_font = os.path.abspath(os.path.join(base_path, "..","data/fonts/NotoSans-Merged.ttf"))

Builder.load_string('''

<TempSpinWait>:
    id: temp_spin
    orientation: 'horizontal'
    adaptive_height: True
    padding: dp(8)

    MDLabel:
        text: "Please wait..."
        font_style: "Subtitle1"
        adaptive_width: True

    MDSpinner:
        size_hint: None, None
        size: dp(14), dp(14)
        active: True

<ChatbotScreen>:
    #on_enter: app.update_chatbot_welcome(self)

    MDBoxLayout: # main box
        orientation: 'vertical'
        padding: 8, 0, 8, 0 # left, top, right, bottom
        spacing: dp(4)

        MDBoxLayout: # top button group
            orientation: 'horizontal'
            adaptive_height: True
            #size_hint_y: 0.1
            spacing: dp(10)

            MDFillRoundFlatIconButton:
                icon: "chat"
                text: "New Chat"
                md_bg_color: '#333036'
                font_size: sp(10)
                on_release: app.new_chat()

            MDDropDownItem:
                #md_bg_color: "#bdc6b0"
                on_release: app.llm_menu.open()
                text: "Model"
                id: llm_menu
                font_size: sp(14)

            MDDropDownItem:
                #md_bg_color: "#bdc6b0"
                on_release: app.token_menu.open()
                text: "Length"
                id: token_menu
                font_size: sp(14)

            Widget:
                size_hint_x: 1

            MDIconButton:
                icon: "menu"
                on_release: app.menu_bar_callback(self)

        MDScrollView: # chat history section with scroll enabled
            size_hint_y: 0.7 # Takes the 70%
            adaptive_height: True

            MDBoxLayout:
                id: chat_history_id
                orientation: 'vertical'
                spacing: dp(10)
                #adaptive_height: True
                size_hint_y: None
                height: self.minimum_height

                #MDLabel:
                #    id: chat_label # chat which will be added

        MDBoxLayout: # Input box
            size_hint_y: 0.2
            orientation: 'horizontal'
            spacing: dp(5)
            padding: 0, 0, 0, 8 # left, top, right, bottom
            adaptive_height: True

            MDTextField:
                id: chat_input
                font_name: root.noto_path
                hint_text: "Ask anyhthing..."
                mode: "rectangle"
                multiline: True
                max_height: "200dp"
                size_hint_x: 0.8
                input_type: 'text'
                keyboard_suggestions: True
                font_size: sp(16)
            MDFillRoundFlatIconButton:
                id: send_msg_button
                icon: "send"
                text: "Go"
                size_hint_x: 0.2
                size_hint_y: 0.8
                font_size: sp(16)
                on_release: app.send_message(self, chat_input)

''')

class TempSpinWait(MDBoxLayout):
    pass

class ChatbotScreen(MDScreen):
    noto_path = StringProperty()
    def __init__(self, noto=noto_font, **kwargs):
        super().__init__(**kwargs)
        self.name = 'chatbot_screen'
        self.noto_path = noto
