from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.list import MDList, OneLineIconListItem, IconLeftWidget, IconRightWidget, OneLineAvatarIconListItem

from kivy.uix.accordion import Accordion, AccordionItem
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.metrics import dp, sp

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

<SettingsBox@MDBoxLayout>:
    orientation: 'vertical'

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
