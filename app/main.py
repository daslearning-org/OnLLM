# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread
import queue
import requests
import time
import datetime
import json
import re

# kivy world
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.core.clipboard import Clipboard
if platform == "android":
    from jnius import autoclass

# kivymd world
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDFloatingActionButton

# app brains
import numpy as np
from onnxruntime import InferenceSession

# other public modules
from m2r2 import convert

# local imports
from myrst import MyRstDocument
from screens.chatbot_screen import TempSpinWait

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

## Global definitions
__version__ = "0.0.1" # The APP version

detect_model_url = "https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
# Determine the base path for your application's resources
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
kv_file_path = os.path.join(base_path, 'main_layout.kv')

## The KivyMD app
class OnLlmApp(MDApp):
    is_downloading = ObjectProperty(None)
    llm_menu = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Window.bind(on_keyboard=self.events)
        self.process = None
        self.sess = None
        self.selected_llm = ""
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        self.top_menu_items = {
            "Demo": {
                "icon": "youtube",
                "action": "web",
                "url": "https://youtube.com/watch?v=a-azvqDL78k",
            },
            "Documentation": {
                "icon": "file-document-check",
                "action": "web",
                "url": "https://blog.daslearning.in/llm_ai/ollama/kivy-chat.html",
            },
            "Contact Us": {
                "icon": "card-account-phone",
                "action": "web",
                "url": "https://daslearning.in/contact/",
            },
            "Check for update": {
                "icon": "github",
                "action": "update",
                "url": "",
            }
        }
        return Builder.load_file(kv_file_path)

    def on_start(self):
        self.llm_models = ["gemma3-1b"]
        if platform == "android":
            # paths on android
            context = autoclass('org.kivy.android.PythonActivity').mActivity
            android_path = context.getExternalFilesDir(None).getAbsolutePath()
            self.model_dir = os.path.join(android_path, 'model_files')
            self.op_dir = os.path.join(android_path, 'outputs')
            config_dir = os.path.join(android_path, 'config')
            self.internal_storage = android_path
            try:
                Environment = autoclass("android.os.Environment")
                self.external_storage = Environment.getExternalStorageDirectory().getAbsolutePath()
            except Exception:
                self.external_storage = os.path.abspath("/storage/emulated/0/")
        else:
            self.internal_storage = os.path.expanduser("~")
            self.external_storage = os.path.expanduser("~")
            self.model_dir = os.path.join(self.user_data_dir, 'model_files')
            config_dir = os.path.join(self.user_data_dir, 'config')
            self.op_dir = os.path.join(self.user_data_dir, 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.op_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        self.gemma_model_path = os.path.join(self.model_dir, "gemma3-1B")
        # hamburger menu
        menu_items = [
            {
                "text": menu_key,
                "leading_icon": self.top_menu_items[menu_key]["icon"],
                "on_release": lambda x=menu_key: self.top_menu_callback(x),
                "font_size": sp(36)
            } for menu_key in self.top_menu_items
        ]
        self.top_menu = MDDropdownMenu(
            items=menu_items,
            width_mult=4,
        )
        self.is_llm_running = False
        ## the chatbot thing
        self.chat_history_id = self.root.ids.chatbot_scr.ids.chat_history_id
        self.chat_history_id.background_color = self.theme_cls.bg_normal
        menu_items = [
            {
                "text": f"{model_name}",
                "leading_icon": "robot-happy",
                "on_release": lambda x=f"{model_name}": self.llm_menu_callback(x),
                "font_size": sp(24)
            } for model_name in self.llm_models
        ]
        self.llm_menu = MDDropdownMenu(
            md_bg_color="#bdc6b0",
            caller=self.root.ids.chatbot_scr.ids.llm_menu,
            items=[],
        )
        if len(self.llm_models) >= 1:
            self.selected_llm = menu_items[0]["text"]
            self.root.ids.chatbot_scr.ids.llm_menu.text = self.selected_llm
            self.llm_menu.items = menu_items
        else:
            # pop up to be added in case of none & disable input
            print("No LLM found!")
            self.llm_menu.items = []
            self.selected_llm = "None"
            self.root.ids.chatbot_scr.ids.llm_menu.text = self.selected_llm
        print("Initialization is successful")

    def show_toast_msg(self, message, is_error=False, duration=3):
        from kivymd.uix.snackbar import MDSnackbar
        bg_color = (0.2, 0.6, 0.2, 1) if not is_error else (0.8, 0.2, 0.2, 1)
        MDSnackbar(
            MDLabel(
                text = message,
                font_style = "Subtitle1" # change size for android
            ),
            md_bg_color=bg_color,
            y=dp(24),
            pos_hint={"center_x": 0.5},
            duration=duration
        ).open()

    def show_text_dialog(self, title, text="", buttons=[]):
        self.txt_dialog = MDDialog(
            title=title,
            text=text,
            buttons=buttons
        )
        self.txt_dialog.open()

    def txt_dialog_closer(self, instance):
        self.txt_dialog.dismiss()

    def menu_bar_callback(self, button):
        self.top_menu.caller = button
        self.top_menu.open()

    def top_menu_callback(self, text_item):
        self.top_menu.dismiss()
        action = ""
        url = ""
        try:
            action = self.top_menu_items[text_item]["action"]
            url = self.top_menu_items[text_item]["url"]
        except Exception as e:
            print(f"Erro in menu process: {e}")
        if action == "web" and url != "":
            self.open_link(url)
        elif action == "update":
            buttons = [
                MDFlatButton(
                    text="Cancel",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_release=self.txt_dialog_closer
                ),
                MDFlatButton(
                    text="Releases",
                    theme_text_color="Custom",
                    text_color="green",
                    on_release=self.update_checker # TBA
                ),
            ]
            self.show_text_dialog(
                "Check for update",
                f"Your version: {__version__}",
                buttons
            )

    def add_bot_message(self, instance, msg_to_add):
        # Adds the Bot msg into chat history
        rst_txt = convert(msg_to_add)
        bot_msg_label = MyRstDocument(
            text = rst_txt,
            base_font_size=36,
            padding=[dp(10), dp(10)],
            background_color = self.theme_cls.bg_normal
        )
        copy_btn = MDFloatingActionButton(
            icon="content-copy",
            type="small",
            theme_icon_color="Custom",
            md_bg_color='#e9dff7',
            icon_color='#211c29',
        )
        copy_btn.bind(on_release=self.copy_rst)
        bot_msg_label.add_widget(copy_btn)
        self.chat_history_id.add_widget(bot_msg_label)

    def copy_rst(self, instance):
        rst_txt = instance.parent.text
        Clipboard.copy(rst_txt)

    def label_copy(self, label_text):
        #print(f"DEBUG: MarkUp Text> {label_text}")
        plain_text = re.sub(r'\[/?(?:color|b|i|u|s|sub|sup|font|font_context|font_family|font_features|size|ref|anchor|text_language).*?\]', '', label_text)
        Clipboard.copy(plain_text)

    def add_usr_message(self, msg_to_add):
        # Adds the User msg into chat history
        usr_msg_label = MDLabel(
            size_hint_y=None,
            markup=True,
            halign='right',
            valign='top',
            padding=[dp(10), dp(10)],
            font_style="Subtitle1",
            allow_selection = True,
            allow_copy = True,
            text = f"{msg_to_add}",
        )
        usr_msg_label.bind(texture_size=usr_msg_label.setter('size'))
        self.chat_history_id.add_widget(usr_msg_label)

    def send_message(self, button_instance, chat_input_widget):
        if self.is_llm_running:
            self.show_toast_msg("Please wait for the current response", is_error=True)
            return
        user_message = chat_input_widget.text.strip()
        if user_message:
            user_message_add = f"[b][color=#2196F3]You:[/color][/b] {user_message}"
            self.messages.append(
                {
                    "role": "user",
                    "content": user_message
                }
            )
            self.add_usr_message(user_message_add)
            chat_input_widget.text = "" # blank the input
            self.tmp_spin = TempSpinWait()
            self.chat_history_id.add_widget(self.tmp_spin)
            ollama_thread = Thread(target=chat_with_llm, args=(self.messages[-3:]), daemon=True) #TBA
            ollama_thread.start()
            self.is_llm_running = True
        else:
            self.show_toast_msg("Please type a message!", is_error=True)

    def llm_menu_callback(self, text_item):
        self.llm_menu.dismiss()
        self.selected_llm = text_item
        self.root.ids.chatbot_scr.ids.llm_menu.text = self.selected_llm

    def update_chatbot_welcome(self, screen_instance):
        print("we are in...")

    def update_checker(self, instance):
        self.txt_dialog.dismiss()
        self.open_link("https://github.com/daslearning-org/OnLLM/releases")

    def open_link(self, url):
        import webbrowser
        webbrowser.open(url)

if __name__ == '__main__':
    OnLlmApp().run()