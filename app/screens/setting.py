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
                MDList:
                    id: delete_model_list
                    #OneLineAvatarIconListItem:
                    #    text: "Preview in Image Selection!"
                    #    IconLeftWidget:
                    #        icon: "robot-happy"
                    #    IconRightWidget:
                    #        icon: "delete"
                    #        on_release: app.img_preview_on()
                    #        theme_text_color: "Custom"
                    #        text_color: "gray"

''')

class DeleteModelItems(OneLineAvatarIconListItem):
    pass

class SettingsBox(MDBoxLayout):
    """ The main settings box which contains the setting, help & other required sections """
