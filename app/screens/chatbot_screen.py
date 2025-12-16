# screens/chatbot_screen.py
import sys, os

from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton, MDFillRoundFlatIconButton, MDFloatingActionButton, MDRectangleFlatIconButton

from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.utils import platform

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
        text: root.text
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
        theme_text_color: "Custom"
        text_color: "black"

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
        theme_text_color: "Custom"
        text_color: "black"

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
        padding: 0, root.top_pad, 0, root.bottom_pad # left, top, right, bottom
        #spacing: dp(4)

        MDBoxLayout: # top button group
            orientation: 'horizontal'
            adaptive_height: True
            #size_hint_y: 0.1
            spacing: dp(10)
            padding: 4, 0, 4, 0
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
                pos_hint: {'center_y': 0.5}
                on_release: app.new_chat()

            MDRectangleFlatIconButton:
                id: llm_menu
                icon: "robot"
                text: "Model"
                theme_text_color: "Custom"
                text_color: "black"
                theme_icon_color: "Custom"
                icon_color: "black"
                line_color: 0, 0, 0, 0
                font_size: sp(10)
                pos_hint: {'center_y': 0.5}
                on_release: app.llm_menu.open()

            MDRectangleFlatIconButton:
                id: token_menu
                icon: "tape-measure"
                text: "Length"
                theme_text_color: "Custom"
                text_color: "black"
                theme_icon_color: "Custom"
                icon_color: "black"
                line_color: 0, 0, 0, 0
                font_size: sp(10)
                pos_hint: {'center_y': 0.5}
                on_release: app.token_menu.open()

            Widget:
                size_hint_x: 1

            MDIconButton:
                icon: "menu"
                theme_text_color: "Custom"
                text_color: "black"
                on_release: app.menu_bar_callback(self)

        MDScrollView: # chat history section with scroll enabled
            size_hint_y: 0.7 # Takes the 70% # 262625: dark
            adaptive_height: True

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

            MDIconButton:
                id: rag_doc
                icon: "file-document-plus"
                icon_size: sp(16)
                pos_hint: {'center_y': 0.5}
                theme_icon_color: "Custom"
                icon_color: "gray"
                on_release: app.rag_file_manager()

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
                fill_color_focus: "#fcfcfc"
                hint_text_color_focus: "gray"

            MDIconButton:
                icon: "send"
                icon_size: sp(24)
                pos_hint: {'center_y': 0.5}
                theme_icon_color: "Custom"
                icon_color: app.theme_cls.primary_color
                on_release: app.send_message(self, chat_input)

''')

class TempSpinWait(MDBoxLayout):
    text = StringProperty("")
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    top_pad = NumericProperty(0)
    bottom_pad = NumericProperty(0)
    def __init__(self, noto=noto_font, **kwargs):
        super().__init__(**kwargs)
        self.name = 'chatbot_screen'
        self.noto_path = noto
        if platform == "android":
            try:
                from android.display_cutout import get_height_of_bar
                self.top_pad = int(get_height_of_bar('status'))
                self.bottom_pad = int(get_height_of_bar('navigation'))
            except Exception as e:
                print(f"Failed android 15 padding: {e}")
                self.top_pad = 32
                self.bottom_pad = 48
