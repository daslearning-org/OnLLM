# python core modules
import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'
import sys
from threading import Thread
import requests
import time
import json
import re

# kivy world
from kivy.lang import Builder
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.core.window import Window
from kivy.metrics import dp, sp
from kivy.utils import platform
from kivy.core.clipboard import Clipboard
from kivy.core.text import LabelBase
from kivy.clock import Clock
from plyer import filechooser
if platform == "android":
    from jnius import autoclass

# kivymd world
from kivymd.app import MDApp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton, MDFloatingActionButton
from kivymd.uix.filemanager import MDFileManager

# app brains
import numpy as np
from onnxruntime import InferenceSession
from tokenizers import Tokenizer

# other public modules
from m2r2 import convert

# local imports
from screens.myrst import MyRstDocument
from screens.chatbot_screen import TempSpinWait, ChatbotScreen, BotResp, BotTmpResp, UsrResp
from screens.welcome import WelcomeScreen
from screens.setting import DeleteModelItems, SettingsBox
from docRag import LocalRag

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

## Global definitions
__version__ = "0.2.0" # The APP version

# Determine the base path for your application's resources
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in a normal Python environment
    base_path = os.path.dirname(os.path.abspath(__file__))
kv_file_path = os.path.join(base_path, 'main_layout.kv')
noto_font = os.path.join(base_path, "data/fonts/NotoSans-Merged.ttf")

## debug if any

## The KivyMD app
class OnLlmApp(MDApp):
    is_downloading = ObjectProperty(None)
    llm_menu = ObjectProperty()
    tmp_txt = ObjectProperty()
    token_count = NumericProperty(128)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #Window.bind(on_keyboard=self.events)
        self.process = None
        self.stop = False
        self.decoder_session = None
        self.rag_sess = None
        self.rag_ok = False
        self.doc_path = None
        self.selected_llm = ""
        self.to_download_model = "na"
        self.messages = []

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        self.top_menu_items = {
            "Settings": {
                "icon": "wrench",
                "action": "settings",
                "url": "",
            },
            "Demo": {
                "icon": "youtube",
                "action": "web",
                "url": "https://www.youtube.com/watch?v=D-KwL59GgKA",
            },
            "Documentation": {
                "icon": "file-document-check",
                "action": "web",
                "url": "https://blog.daslearning.in/llm_ai/genai/onllm.html",
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
            },
            "Try Other Apps": {
                "icon": "google-play",
                "action": "web",
                "url": "https://daslearning.in/apps/",
            },
        }
        return Builder.load_file(kv_file_path)

    def on_start(self):
        self.llm_models = {
            "smollm2-135m": {
                "name": "smollm2-135m",
                "url": "https://github.com/daslearning-org/OnLLM/releases/download/vOnnxModels/smollm2-135m.tar.gz",
                "size": "95MB",
                "platform": "android", # means runs on all
                "tokens": ["", "<|im_start|>", "<|im_end|>"],
                "eos_ids": ["<|endoftext|>"],
                "att_mask": True
            }
        }
        self.rag_models = {
            "all-MiniLM-L6-V2": {
                "name": "all-MiniLM-L6-V2",
                "url": "https://huggingface.co/daslearning/Embedding-Onnx/resolve/main/onnx/all-MiniLM-L6-V2.tar.gz?download=true",
                "size": "85MB",
                "platform": "android"
            }
        }
        if platform == "android":
            from android.permissions import request_permissions, Permission
            sdk_version = 28
            try:
                VERSION = autoclass('android.os.Build$VERSION')
                sdk_version = VERSION.SDK_INT
                print(f"Android SDK: {sdk_version}")
                #self.show_toast_msg(f"Android SDK: {sdk_version}")
            except Exception as e:
                print(f"Could not check the android SDK version: {e}")
            if sdk_version >= 33:  # Android 13+
                permissions = [Permission.READ_EXTERNAL_STORAGE]
            else:  # Android 9–12
                permissions = [Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE]
            request_permissions(permissions)
            # paths on android
            context = autoclass('org.kivy.android.PythonActivity').mActivity
            android_path = context.getExternalFilesDir(None).getAbsolutePath()
            self.model_dir = os.path.join(android_path, 'model_files')
            self.op_dir = os.path.join(android_path, 'outputs')
            self.config_dir = os.path.join(android_path, 'config')
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
            self.config_dir = os.path.join(self.user_data_dir, 'config')
            self.op_dir = os.path.join(self.user_data_dir, 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.op_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        # update models from local model config
        self.extra_models_config = os.path.join(self.config_dir, 'extra_models.json')
        if os.path.exists(self.extra_models_config):
            with open(self.extra_models_config, "r") as modelfile:
                model_json_obj = json.load(modelfile)
            for model in model_json_obj:
                if platform == "android":
                    if model_json_obj[model]['platform'] in ("android", "warn"):
                        self.llm_models[model] = model_json_obj[model]
                else:
                    self.llm_models[model] = model_json_obj[model]
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
        )
        self.is_llm_running = False
        ## the chatbot thing
        self.chat_history_id = self.root.ids.chatbot_scr.ids.chat_history_id
        self.chat_history_id.background_color = self.theme_cls.bg_normal
        # llm models drop-down
        self.set_llm_dropdown(stage="init")
        # token drop-down
        token_sizes = [128, 256, 512, 1024, 2048]
        token_drop_items = [
            {
                "text": f"{tkn_size}",
                "on_release": lambda x=f"{tkn_size}": self.token_menu_callback(x),
                "font_size": sp(24)
            } for tkn_size in token_sizes
        ]
        # token size menu
        self.token_menu = MDDropdownMenu(
            md_bg_color="#bdc6b0",
            caller=self.root.ids.chatbot_scr.ids.token_menu,
            items=token_drop_items,
        )
        self.root.ids.chatbot_scr.ids.token_menu.text = str(token_sizes[0])
        self.is_doc_manager_open = False
        print("Initialisation is successful")

    def doc_file_exit_manager(self, instance=None):
        self.is_doc_manager_open = False

    def select_doc_path(self, path):
        self.doc_file_exit_manager()
        if path:
            Clock.schedule_once(lambda dt: self.doc_selection_processing(path))

    def doc_selection_processing(self, path):
        if path:
            self.doc_path = str(path[0])
            print(f"\n**Selected doc path: {self.doc_path} & full path list {path}") # debug
            self.tmp_wait = TempSpinWait()
            self.tmp_wait.text = "Analyzing the doc, please wait..."
            self.chat_history_id.add_widget(self.tmp_wait)
            if not self.rag_sess:
                self.rag_sess = LocalRag(
                    model_dir=self.model_dir,
                    config_dir=self.config_dir
                )
            Thread(target=self.rag_sess.start_rag_onnx_sess, args=(self.doc_path, self.rag_init_callback), daemon=True).start()

    def rag_file_manager(self):
        """Open the file manager to select a doc file. On android use Downloads or Documents folders only"""
        rag_btn = self.root.ids.chatbot_scr.ids.rag_doc
        if self.rag_ok:
            self.rag_ok = False
            rag_btn.icon = "file-document-plus"
            rag_btn.icon_color = "gray"
        else:
            model_name = "all-MiniLM-L6-V2"
            rag_model_exists = self.check_rag_models(model_name)
            if self.is_downloading:
                self.show_toast_msg("Please wait for the current download to be finished!", is_error=True)
                return
            if not rag_model_exists:
                llm_size = self.rag_models[model_name]['size']
                self.to_download_model = model_name
                self.model_file_size = f"You need to downlaod the file for the first time (~{llm_size})"
                self.popup_download_model()
                #self.show_toast_msg("You need to dowload the model first!") # apply actual logic with popup
                return
            try:
                filechooser.open_file(
                    on_selection=self.select_doc_path,
                    filters=[
                        "*.pdf",
                        "*.docx"
                    ]
                )
                self.is_doc_manager_open = True
            except Exception as e:
                self.show_toast_msg(f"Error: {e}", is_error=True)

    def set_llm_dropdown(self, stage="post-init"):
        menu_items = []
        # llm model drop-down items
        for model in self.llm_models:
            model_name = model
            tmp_menu = {
                "text": f"{model_name}",
                "leading_icon": "robot-happy",
                "on_release": lambda x=f"{model_name}": self.llm_menu_callback(x),
                "font_size": sp(24)
            }
            menu_items.append(tmp_menu)
        # model menu
        self.llm_menu = MDDropdownMenu(
            md_bg_color="#bdc6b0",
            caller=self.root.ids.chatbot_scr.ids.llm_menu,
            items=[],
        )
        self.llm_menu.items = menu_items
        if stage == "init" or self.selected_llm == "":
            self.root.ids.chatbot_scr.ids.llm_menu.text = "Select model"
        else:
            self.root.ids.chatbot_scr.ids.llm_menu.text = self.selected_llm

    def start_from_welcome(self):
        Thread(target=self.model_sync_on_init, args=("main",), daemon=True).start()
        self.root.current = "chatbot_screen"

    def check_model_files(self, model_name):
        path_to_model = os.path.join(self.model_dir, f"{model_name}")
        model_config = os.path.join(path_to_model, "config.json")
        model_tokenizer = os.path.join(path_to_model, "tokenizer.json")
        model_onnx = os.path.join(path_to_model, "onnx", "model_int8.onnx")
        if not os.path.exists(model_config) or not os.path.exists(model_tokenizer) or not os.path.exists(model_onnx):
            return False
        else:
            return True

    def check_rag_models(self, model_name):
        path_to_model = os.path.join(self.model_dir, f"{model_name}")
        model_tokenizer = os.path.join(path_to_model, "tokenizer.json")
        model_onnx = os.path.join(path_to_model, "model.onnx")
        if not os.path.exists(model_tokenizer) or not os.path.exists(model_onnx):
            return False
        else:
            return True

    def model_sync_on_init(self, branch="develop"):
        url = f"https://raw.githubusercontent.com/daslearning-org/OnLLM/{branch}/app/extra_models.json"
        url += f"?_t={int(time.time())}"  # prevents CDN caching
        filename = url.split("/")[-1]
        flag = False
        try:
            import certifi
            response = requests.get(url, verify=certifi.where(), timeout=10)
            response.raise_for_status()
            with open(self.extra_models_config, "wb") as f:
                f.write(response.content)
            if os.path.exists(self.extra_models_config):
                with open(self.extra_models_config, "r") as modelfile:
                    model_json_obj = json.load(modelfile)
                for model in model_json_obj:
                    if (not model in self.llm_models):
                        if platform == "android":
                            if model_json_obj[model]['platform'] == "android":
                                self.llm_models[model] = model_json_obj[model]
                                flag = True
                        else:
                            self.llm_models[model] = model_json_obj[model]
                            flag = True
        except Exception as e:
            print(f"Cannot get the extra models json from GitHub: {e}")
        if flag:
            Clock.schedule_once(lambda dt: self.set_llm_dropdown())

    def popup_download_model(self):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.txt_dialog_closer
            ),
            MDFlatButton(
                text="Ok",
                theme_text_color="Custom",
                text_color="green",
                on_release=self.initiate_model_download
            ),
        ]
        self.show_text_dialog(
            "Downlaod the model file",
            self.model_file_size,
            buttons
        )

    def update_download_progress(self, downloaded, total_size):
        if total_size > 0:
            percentage = (downloaded / total_size) * 100
            self.download_progress.text = f"Progress: {percentage:.1f}%"
        else:
            self.download_progress.text = f"Progress: {downloaded} bytes"

    def download_file(self, download_url, download_path):
        filename = download_url.split("/")[-1]
        filename = filename.replace("?download=true", "")
        try:
            self.is_downloading = filename
            with requests.get(download_url, stream=True) as req:
                req.raise_for_status()
                total_size = int(req.headers.get('content-length', 0))
                downloaded = 0
                with open(download_path, 'wb') as f:
                    for chunk in req.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            Clock.schedule_once(lambda dt: self.update_download_progress(downloaded, total_size))
            if os.path.exists(download_path):
                Clock.schedule_once(lambda dt: self.unzip_model(download_path))
            else:
                Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the onnx file: {e} 😞")
            Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
        self.is_downloading = False
        self.to_download_model = "na"

    def download_model_file(self, model_url, download_path, instance=None):
        self.txt_dialog_closer(instance)
        filename = download_path.split("/")[-1]
        print(f"Starting the download for: {filename}")
        Thread(target=self.download_file, args=(model_url, download_path), daemon=True).start()

    def initiate_model_download(self, instance):
        if self.to_download_model == "na":
            return
        self.download_progress = MDLabel(
            text = "Starting download process...",
            font_style = "Subtitle1",
            halign = 'left',
            adaptive_height = True,
            theme_text_color = "Custom",
            text_color = "#f7f7f5"
        )
        self.chat_history_id.add_widget(self.download_progress)
        model_name = self.to_download_model
        embed_model_list = list(self.rag_models.keys())
        if model_name in embed_model_list:
            url = self.rag_models[model_name]['url']
        else:
            url = self.llm_models[model_name]['url']
        path_to_model = os.path.join(self.model_dir, f"{model_name}.tar.gz")
        self.download_model_file(url, path_to_model, instance)

    def unzip_model(self, filepath):
        import tarfile
        try:
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=self.model_dir)
            os.remove(filepath)
            self.show_toast_msg("Model has been downloaded successfully.")
            self.is_downloading = False
        except Exception as e:
            print(f"Unzip error: {e}")

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

    def txt_dialog_closer(self, instance=None):
        if self.txt_dialog:
            self.txt_dialog.dismiss()

    def show_custom_dialog(self, title, content_cls, buttons=[]):
        self.cst_dialog = MDDialog(
            title=title,
            type="custom",
            content_cls=content_cls,
            buttons=buttons
        )
        self.txt_dialog.open()

    def custom_dialog_closer(self, instance):
        if self.cst_dialog:
            self.cst_dialog.dismiss()

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
                    on_release=self.update_checker
                ),
            ]
            self.show_text_dialog(
                "Check for update",
                f"Your version: {__version__}",
                buttons
            )
        elif action == "settings":
            self.root.current = "settings_screen"

    def llm_menu_callback(self, text_item):
        self.llm_menu.dismiss()
        check_model = self.check_model_files(text_item)
        llm_size = self.llm_models[text_item]['size']
        if not check_model:
            if self.is_downloading:
                self.show_toast_msg("Please wait for the current download to finish", is_error=True, duration=2)
                return
            self.to_download_model = text_item
            self.model_file_size = f"You need to downlaod the file for the first time (~{llm_size})"
            self.popup_download_model()
            return
        self.selected_llm = text_item
        self.root.ids.chatbot_scr.ids.llm_menu.text = self.selected_llm
        self.init_onnx_sess(self.selected_llm)

    def token_menu_callback(self, text):
        self.token_menu.dismiss()
        self.token_count = int(text)
        self.root.ids.chatbot_scr.ids.token_menu.text = text

    def rag_init_callback(self, check):
        rag_btn = self.root.ids.chatbot_scr.ids.rag_doc
        if check:
            self.rag_ok = True
            rag_btn.icon = "file-document-remove"
            rag_btn.icon_color = "orange"
            self.show_toast_msg("Document processed, you can ask quesions on your DOC")
        else:
            self.show_toast_msg("Document processed failed, your answer will be generic", is_error=True)
        if self.tmp_wait:
            self.chat_history_id.remove_widget(self.tmp_wait)
            self.tmp_wait = None

    def rag_qa_callback(self, prompt):
        self.send_message(button_instance=None, chat_input_widget=None, callback=True, rag_usr_prompt=prompt)

    def stop_chat(self):
        self.stop = True
        self.is_llm_running = False

    def new_chat(self):
        self.stop = True
        self.is_llm_running = False
        self.chat_history_id.clear_widgets()
        self.messages = []

    def add_bot_message(self, msg_to_add, msg_id):
        # Adds the Bot msg into chat history
        rst_txt = convert(msg_to_add)
        bot_msg_label = BotResp()
        bot_msg_label.text = rst_txt
        bot_msg_label.given_id = msg_id
        self.chat_history_id.add_widget(bot_msg_label)

    def copy_tmp_msg(self, instance):
        rst_txt = instance.parent.parent.text
        Clipboard.copy(rst_txt)

    def copy_final_msg(self, instance):
        given_id = int(instance.parent.parent.given_id)
        if given_id == 999:
            rst_txt = instance.parent.parent.text
        else:
            rst_txt = str(self.messages[given_id]["content"])
        Clipboard.copy(rst_txt)

    def label_copy(self, label_text):
        #print(f"DEBUG: MarkUp Text> {label_text}")
        plain_text = re.sub(r'\[/?(?:color|b|i|u|s|sub|sup|font|font_context|font_family|font_features|size|ref|anchor|text_language).*?\]', '', label_text)
        Clipboard.copy(plain_text)

    def add_usr_message(self, msg_to_add):
        # Adds the User msg into chat history
        usr_msg_label = UsrResp()
        usr_msg_label.text = msg_to_add
        self.chat_history_id.add_widget(usr_msg_label)

    def send_message(self, button_instance, chat_input_widget, callback=False, rag_usr_prompt=""):
        if self.selected_llm == "":
            self.show_toast_msg("Please select a model first!", is_error=True)
            return
        if self.is_llm_running:
            self.show_toast_msg("Please wait for the current response", is_error=True)
            return
        if callback:
            user_message = rag_usr_prompt.strip()
            llm_context = {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."}
            if self.tmp_wait:
                self.chat_history_id.remove_widget(self.tmp_wait)
                self.tmp_wait = None
        else:
            user_message = chat_input_widget.text.strip()
            if self.rag_ok:
                Thread(target=self.rag_sess.get_rag_prompt, args=(user_message,self.rag_qa_callback), daemon=True).start()
                self.tmp_wait = TempSpinWait()
                self.tmp_wait.text = "Please wait while reading the doc..."
                self.chat_history_id.add_widget(self.tmp_wait)
                #return
            llm_context = {"role": "system", "content": "You are a helpful assistant."}
            chat_input_widget.text = ""
        if user_message:
            user_message_add = f"{user_message}"
            if not callback:
                self.add_usr_message(user_message_add)
                if self.rag_ok:
                    # RAG will use the callback method
                    return
                self.messages.append(
                    {
                        "role": "user",
                        "content": user_message
                    }
                )
            self.tmp_txt = BotTmpResp()
            self.chat_history_id.add_widget(self.tmp_txt)
            msg_to_send = [llm_context]
            if callback:
                rag_msg = {
                    "role": "user",
                    "content": user_message
                }
                msg_to_send.append(rag_msg)
            else:
                msg_to_send.extend(self.messages[-3:]) # taking last three messages only
            ollama_thread = Thread(target=self.chat_with_llm, args=(msg_to_send,), daemon=True)
            ollama_thread.start()
            self.is_llm_running = True
        else:
            self.show_toast_msg("Please type a message!", is_error=True)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_tensors="np"):
        init_prompt = self.llm_models[self.selected_llm]['tokens'][0]
        start_prompt = self.llm_models[self.selected_llm]['tokens'][1]
        end_prompt = self.llm_models[self.selected_llm]['tokens'][2]
        prompt = init_prompt
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()  # Strip for cleanliness
            if role == "system":
                prompt += f"{start_prompt}system\n{content}{end_prompt}\n"
            elif role == "user":
                prompt += f"{start_prompt}user\n{content}{end_prompt}\n"
            elif role == "assistant" or role == "model":
                prompt += f"{start_prompt}assistant\n{content}{end_prompt}\n"

        if add_generation_prompt:
            prompt += f"{start_prompt}assistant\n"

        if not tokenize:
            return prompt

        # Tokenize (encode to IDs)
        encoding = self.tokenizer.encode(prompt, add_special_tokens=False)  # False to avoid extra BOS if already added
        token_ids = encoding.ids

        if return_tensors == "np":
            input_ids = np.array([token_ids], dtype=np.int64)  # Batch size 1
        else:
            input_ids = np.array(token_ids, dtype=np.int64)

        # Return dict like HF (only input_ids, as in your code)
        return {"input_ids": input_ids}

    def init_onnx_sess(self, llm="smollm2-135m"):
        path_to_model = os.path.join(self.model_dir, llm)
        arm_android = False
        try:
            import platform as corept
            cpu_arch = corept.machine()
            if 'arm' in cpu_arch.lower() or 'aarch' in cpu_arch.lower():
                arm_android = True
        except Exception as e:
            print(f"Error in CPU architecture check: {e}")
        android_providers = [
            'XnnpackExecutionProvider',
            'CPUExecutionProvider',
        ]
        desktop_providers = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        try:
            # Load config & token jsons
            # Load config with json
            with open(f"{path_to_model}/config.json", "r") as f:
                config_data = json.load(f)
            self.tokenizer = Tokenizer.from_file(f"{path_to_model}/tokenizer.json")
            self.num_key_value_heads = config_data["num_key_value_heads"]
            self.head_dim = config_data["head_dim"]
            self.num_hidden_layers = config_data["num_hidden_layers"]
            primary_eos = self.llm_models[self.selected_llm]["tokens"][2]
            other_eos = self.llm_models[self.selected_llm]["eos_ids"]
            self.eos_token_ids = [
                self.tokenizer.token_to_id(primary_eos),
            ]
            for eos in other_eos:
                eos_item = self.tokenizer.token_to_id(str(eos))
                self.eos_token_ids.append(eos_item)

            if platform == "android" or arm_android:
                self.decoder_session = InferenceSession(f"{path_to_model}/onnx/model_int8.onnx", providers=android_providers)
            else:
                self.decoder_session = InferenceSession(f"{path_to_model}/onnx/model_int8.onnx", providers=desktop_providers)
            print("Using:", self.decoder_session.get_providers())
            self.process = True
        except Exception as e:
            print(f"Onnx init error: {e}")
            self.show_toast_msg(f"Onnx init error: {e}", is_error=True)

    def sample_logits(self, logits, temperature=0.7, top_p=0.9):
        logits = logits.astype(np.float64)
        logits = logits / max(temperature, 1e-5)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        sorted_indices = np.argsort(probs[0])[::-1]
        sorted_probs = probs[0, sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.where(cumulative_probs > top_p)[0]
        cutoff = cutoff[0] + 1 if len(cutoff) > 0 else len(probs[0])
        probs[:, sorted_indices[cutoff:]] = 0
        probs /= np.sum(probs, axis=-1, keepdims=True) + 1e-10
        next_token = np.random.choice(len(probs[0]), p=probs[0])
        return np.array([[next_token]])

    def chat_with_llm(self, messages):
        if not self.process:
            self.is_llm_running = False
            Clock.schedule_once(lambda dt: self.show_toast_msg("Onnx Session is not ready", is_error=True))
            return
        # start onnx llm inference
        self.stop = False
        final_result = {"role": "init", "content": "Chat initial"}
        final_txt = ""
        try:
            inputs = self.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="np")
            input_token_count = inputs['input_ids'].shape[-1]
            print(f"Input token count: {input_token_count}")
            ## Prepare decoder inputs
            input_ids = inputs['input_ids']
            batch_size = inputs['input_ids'].shape[0]
            past_key_values = {
                f'past_key_values.{layer}.{kv}': np.zeros(
                    [batch_size, self.num_key_value_heads, 1, self.head_dim],
                    dtype=np.float32
                )[:, :, :0, :]  # Slice back to 0-length safely
                for layer in range(self.num_hidden_layers)
                for kv in ('key', 'value')
            }
            use_att_mask = self.llm_models[self.selected_llm].get("att_mask", False)
            if use_att_mask:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
            position_ids = np.tile(np.arange(0, input_ids.shape[-1]), (batch_size, 1))
            max_new_tokens = int(self.token_count)
            #generated_tokens = input_ids
            for i in range(max_new_tokens):
                if use_att_mask:
                    logits, *present_key_values = self.decoder_session.run(None, dict(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **past_key_values,
                    ))
                else:
                    logits, *present_key_values = self.decoder_session.run(None, dict(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        **past_key_values,
                    ))

                ## Update values for next generation loop
                #input_ids = np.argmax(logits[:, -1], axis=-1, keepdims=True)
                input_ids = self.sample_logits(logits[:, -1, :], temperature=0.7, top_p=0.9)
                if use_att_mask:
                    attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids, dtype=np.int64)], axis=-1)
                position_ids = position_ids[:, -1:] + 1
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

                #generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
                if np.isin(input_ids, self.eos_token_ids).any() or self.stop:
                    break

                ## (Optional) Streaming (use tokenizer.decode)
                txt_update = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                if txt_update and not self.stop:
                    final_txt += str(txt_update)
                    Clock.schedule_once(lambda dt: self.update_text_stream(txt_update))
            # final result
            final_result["content"] = final_txt
            final_result["role"] = "assistant"
        except Exception as e:
            print(f"Chat error: {e}")
            final_result["content"] = f"**Error** with LLM: {e}"
            final_result["role"] = "error"
        if not self.stop:
            Clock.schedule_once(lambda dt: self.final_llm_result(final_result))

    def popup_delete_model(self, model=""):
        buttons = [
            MDFlatButton(
                text="Cancel",
                theme_text_color="Custom",
                text_color=self.theme_cls.primary_color,
                on_release=self.cancel_delete_model
            ),
            MDFlatButton(
                text="Delete",
                theme_text_color="Custom",
                text_color="red",
                on_release=self.delete_model_confirm
            ),
        ]
        self.show_text_dialog(
            "Delete the model?",
            f"{model} will be deleted & action cannot be undone!",
            buttons
        )

    def delete_model_confirm(self, instance):
        if self.to_delete_model:
            delete_path = os.path.join(self.model_dir, self.to_delete_model)
            try:
                import shutil
                shutil.rmtree(delete_path)
                self.show_toast_msg(f"Deleted all files of {self.to_delete_model}")
            except Exception as e:
                self.show_toast_msg(f"Could not delete due to: {e}", is_error=True)
        self.to_delete_model = False
        self.txt_dialog_closer()

    def init_delete_model(self, model):
        """ Call back from delete icon """
        self.to_delete_model = model
        self.popup_delete_model(model)

    def cancel_delete_model(self, instance):
        self.txt_dialog_closer()
        self.to_delete_model = False

    def settings_initiate(self):
        set_scroll = self.root.ids.settings_scr.ids.delete_model_list
        set_scroll.clear_widgets()
        model_folders = os.listdir(self.model_dir)
        for model in model_folders:
            tmp_list_item = DeleteModelItems()
            tmp_list_item.text = model
            set_scroll.add_widget(tmp_list_item)

    def go_to_chat_screen(self):
        self.root.current = "chatbot_screen"

    def update_text_stream(self, txt_update):
        if self.tmp_txt:
            self.tmp_txt.text = self.tmp_txt.text + txt_update

    def final_llm_result(self, llm_resp):
        if llm_resp["role"] == "assistant":
            self.messages.append(llm_resp)
            msg_id = len(self.messages) - 1
        else:
            msg_id = 999
        self.is_llm_running = False
        txt = llm_resp["content"]
        self.chat_history_id.remove_widget(self.tmp_txt)
        self.add_bot_message(msg_to_add=txt, msg_id=msg_id)

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