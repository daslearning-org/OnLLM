from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.list import MDList, OneLineIconListItem, IconLeftWidget, IconRightWidget, OneLineAvatarIconListItem

from kivy.uix.accordion import Accordion, AccordionItem
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.metrics import dp, sp
from kivy.utils import platform

# local imports

Builder.load_string('''

<DeleteModelItems@OneLineAvatarIconListItem>:
    IconLeftWidget:
        icon: "robot-happy"
    IconRightWidget:
        icon: "delete"
        on_release: app.init_delete_model(root.text)
        theme_text_color: "Custom"
        text_color: "gray"

<SettingsBox>:
    orientation: 'vertical'
    padding: 0, root.top_pad, 0, root.bottom_pad

    Accordion:
        orientation: 'vertical'

        AccordionItem:
            title: "Delete model files"
            spacing: dp(8)
            canvas.before:
                Color:
                    rgba: 168, 183, 191, 1
                RoundedRectangle:
                    size: self.width, self.height
                    pos: self.pos

            MDScrollView:
                adaptive_height: True
                MDList:
                    id: delete_model_list
                    # Items will be added here

    MDBoxLayout: # Input box with Send button
        size_hint_y: 0.1
        orientation: 'horizontal'
        spacing: dp(5)
        padding: 8, 4, 8, 8 # left, top, right, bottom
        adaptive_height: True
        MDFillRoundFlatIconButton:
            id: setting_to_chat
            icon: "chat"
            text: "Go Back"
            size_hint_x: 0.2
            font_size: sp(16)
            on_release: app.go_to_chat_screen()

''')

class DeleteModelItems(OneLineAvatarIconListItem):
    pass

class SettingsBox(MDBoxLayout):
    """ The main settings box which contains the setting, help & other required sections """
    top_pad = NumericProperty(0)
    bottom_pad = NumericProperty(0)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if platform == "android":
            try:
                from android.display_cutout import get_height_of_bar
                self.top_pad = int(get_height_of_bar('status'))
                self.bottom_pad = int(get_height_of_bar('navigation'))
            except Exception as e:
                print(f"Failed android 15 padding: {e}")
                self.top_pad = 32
                self.bottom_pad = 48
