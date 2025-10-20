# screens/chatbot_screen.py
import sys, os

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton, MDFillRoundFlatIconButton, MDFloatingActionButton

from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ObjectProperty

# local imports
from .myrst import MyRstDocument

# get path details
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
    noto_font = os.path.join(base_path, "data/fonts/NotoSans-Merged.ttf")
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
    noto_font = os.path.abspath(os.path.join(base_path, "..", "data/fonts/NotoSans-Merged.ttf"))

Builder.load_string('''
#:import parse_color kivy.parser.parse_color

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

<UsrResp>:
    orientation: 'vertical'
    size_hint_y: None
    height: content.texture_size[1] + dp(20)
    size_hint_x: None
    size_hint_x: 0.6
    pos_hint: {"right": 1}
    padding: 4, 2
    spacing: dp(2)
    canvas.before:
        Color:
            rgb: parse_color('#c0d9bd')
        RoundedRectangle:
            size: self.width, self.height
            pos: self.pos
            radius: [20, 20, 0, 20]

    MDLabel:
        id: content
        font_style: "Subtitle1"
        text: root.text
        markup: True
        #valign: 'top'
        halign: 'right'
        allow_selection: True,
        allow_copy: True,
        padding: 8

<BotResp>:
    orientation: 'vertical'
    size_hint_y: None
    size_hint_x: 0.9
    pos_hint: {"x": 0}
    height: self.minimum_height + dp(10)
    padding: 4, 2
    spacing: dp(2)
    canvas.before:
        Color:
            rgb: parse_color('#eddbb7')
        RoundedRectangle:
            size: self.width, self.height
            pos: self.pos
            radius: [20, 20, 20, 0]

    MyRstDocument:
        base_font_size: 36
        padding: 8
        text: root.text
        background_color: parse_color('#eddbb7')

    MDBoxLayout:
        orientation: 'horizontal'
        padding: dp(4)
        size_hint_y: None
        height: self.minimum_height
        Widget:
            size_hint_x: 1
        MDFloatingActionButton:
            icon: 'content-copy'
            type: 'small'
            theme_icon_color: "Custom"
            md_bg_color: '#e9dff7'
            icon_color: '#211c29'
            on_release: app.copy_final_msg(self)

<BotTmpResp>:
    orientation: 'vertical'
    size_hint_y: None
    height: content.texture_size[1] + dp(32)
    size_hint_x: 0.9
    pos_hint: {"x": 0}
    padding: 4, 2
    spacing: dp(2)
    canvas.before:
        Color:
            rgb: parse_color('#eddbb7')
        RoundedRectangle:
            size: self.width, self.height
            pos: self.pos
            radius: [20, 20, 20, 0]

    MDLabel:
        id: content
        font_style: "Subtitle1"
        text: root.text
        valign: 'top'
        halign: 'left'
        padding: 8

    MDBoxLayout:
        orientation: 'horizontal'
        spacing: dp(4)
        size_hint_y: None
        height: self.minimum_height
        Widget:
            size_hint_x: 1
        MDFloatingActionButton:
            icon: 'content-copy'
            type: 'small'
            theme_icon_color: "Custom"
            md_bg_color: '#e9dff7'
            icon_color: '#211c29'
            on_release: app.copy_tmp_msg(self)
        MDFloatingActionButton:
            icon: 'stop'
            type: 'small'
            theme_icon_color: "Custom"
            md_bg_color: '#e9dff7'
            icon_color: '#211c29'
            on_release: app.stop_chat()

<ChatbotScreen>:
    #on_enter: app.update_chatbot_welcome(self)

    MDBoxLayout: # main box
        orientation: 'vertical'
        #padding: 8, 0, 8, 0 # left, top, right, bottom
        #spacing: dp(4)

        MDBoxLayout: # top button group
            orientation: 'horizontal'
            adaptive_height: True
            #size_hint_y: 0.1
            spacing: dp(10)
            padding: 4, 0, 4, 8
            canvas.before:
                Color:
                    rgb: parse_color('#dfcaeb')
                Rectangle:
                    size: self.width, self.height
                    pos: self.pos

            MDFillRoundFlatIconButton:
                icon: "chat"
                text: "New"
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
            canvas.before:
                Color:
                    rgb: parse_color('#262625')
                Rectangle:
                    size: self.width, self.height
                    pos: self.pos

            # all chats will be added under this box
            MDBoxLayout:
                id: chat_history_id
                padding: dp(4)
                orientation: 'vertical'
                spacing: dp(10)
                size_hint_y: None
                height: self.minimum_height

        MDBoxLayout: # Input box with Send button
            size_hint_y: 0.2
            orientation: 'horizontal'
            spacing: dp(5)
            padding: 8, 4, 8, 8 # left, top, right, bottom
            adaptive_height: True
            canvas.before:
                Color:
                    rgb: parse_color('#262625')
                Rectangle:
                    size: self.width, self.height
                    pos: self.pos

            MDTextField:
                id: chat_input
                font_name: root.noto_path
                hint_text: "Ask anyhthing"
                mode: "fill" #"rectangle"
                multiline: True
                max_height: "200dp"
                size_hint_x: 0.8
                input_type: 'text'
                keyboard_suggestions: True
                font_size: sp(16)

            MDIconButton:
                icon: "send"
                icon_size: sp(32)
                pos_hint: {'center_y': 0.5}
                theme_icon_color: "Custom"
                icon_color: app.theme_cls.primary_color
                on_release: app.send_message(self, chat_input)

''')

class TempSpinWait(MDBoxLayout):
    pass

class UsrResp(MDBoxLayout):
    text = StringProperty("")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BotTmpResp(MDBoxLayout):
    text = StringProperty("")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BotResp(MDBoxLayout):
    text = StringProperty("")
    given_id = NumericProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ChatbotScreen(MDScreen):
    noto_path = StringProperty()
    def __init__(self, noto=noto_font, **kwargs):
        super().__init__(**kwargs)
        self.name = 'chatbot_screen'
        self.noto_path = noto
