# screens/chatbot_screen.py
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dropdownitem import MDDropDownItem
from kivymd.uix.button import MDIconButton, MDFillRoundFlatButton

from kivy.uix.widget import Widget
from kivy.lang import Builder
from kivy.metrics import dp, sp

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
        padding: dp(8)
        spacing: dp(10)

        MDBoxLayout: # top button group
            orientation: 'horizontal'
            adaptive_height: True
            #size_hint_y: 0.1
            spacing: dp(10)

            MDDropDownItem:
                md_bg_color: "#bdc6b0"
                #pos_hint: {"center_x": .5, "center_y": .7}
                on_release: app.llm_menu.open()
                text: "Choose Model"
                id: llm_menu
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
            adaptive_height: True

            MDTextField:
                id: chat_input
                hint_text: "Ask anyhthing..."
                mode: "rectangle"
                multiline: True
                max_height: "200dp"
                size_hint_x: 0.8
                input_type: 'text'
                keyboard_suggestions: True
                font_size: sp(16)
            MDFillRoundFlatButton:
                id: send_msg_button
                text: "Send"
                size_hint_x: 0.2
                size_hint_y: 1
                font_size: sp(16)
                on_release: app.send_message(self, chat_input)

''')

class TempSpinWait(MDBoxLayout):
    pass

class ChatbotScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'chatbot_screen'
