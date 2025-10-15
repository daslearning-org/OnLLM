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
from kivy.core.text import LabelBase
from kivy.clock import Clock
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
from tokenizers import Tokenizer

# other public modules
from m2r2 import convert

# local imports
from screens.myrst import MyRstDocument
from screens.chatbot_screen import TempSpinWait, ChatbotScreen, BotResp, BotTmpResp, UsrResp
from screens.welcome import WelcomeScreen

# IMPORTANT: Set this property for keyboard behavior
Window.softinput_mode = "below_target"

## Global definitions
__version__ = "0.0.2" # The APP version

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
        self.selected_llm = ""
        self.messages = []

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Orange"
        self.top_menu_items = {
            "Demo": {
                "icon": "youtube",
                "action": "web",
                "url": "https://www.youtube.com/playlist?list=PL7ZAVvBwIkXYJPUA3Wvkykk0u7DYWO3OI", # TBA
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
        self.llm_models = [
            {
                "name": "smollm2-135m",
                "url": "https://github.com/daslearning-org/OnLLM/releases/download/vOnnxModels/smollm2-135m.tar.gz"
            }
        ]
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
        menu_items = []
        for model in self.llm_models:
            model_name = model["name"]
            tmp_menu = {
                "text": f"{model_name}",
                "leading_icon": "robot-happy",
                "on_release": lambda x=f"{model_name}": self.llm_menu_callback(x),
                "font_size": sp(24)
            }
            menu_items.append(tmp_menu)
        token_sizes = [128, 256, 512, 1024]
        token_drop_items = [
            {
                "text": f"{tkn_size}",
                "leading_icon": "robot-happy",
                "on_release": lambda x=f"{tkn_size}": self.token_menu_callback(x),
                "font_size": sp(24)
            } for tkn_size in token_sizes
        ]
        # model menu
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
        # token size menu
        self.token_menu = MDDropdownMenu(
            md_bg_color="#bdc6b0",
            caller=self.root.ids.chatbot_scr.ids.token_menu,
            items=token_drop_items,
        )
        self.root.ids.chatbot_scr.ids.token_menu.text = str(token_sizes[0])
        print("Initialisation is successful")

    def start_from_welcome(self):
        model_name = self.llm_models[0]['name']
        path_to_model = os.path.join(self.model_dir, f"{model_name}")
        model_config = os.path.join(path_to_model, "config.json")
        model_tokenizer = os.path.join(path_to_model, "tokenizer.json")
        model_onnx = os.path.join(path_to_model, "onnx", "model_int8.onnx")
        if self.is_downloading:
            self.show_toast_msg("Please wait for the downlaod to complete!", is_error=True)
            return
        if not os.path.exists(model_config) or not os.path.exists(model_tokenizer) or not os.path.exists(model_onnx):
            self.popup_smol135m_model()
            return
        self.init_onnx_sess()
        self.root.current = "chatbot_screen"

    def stop_chat(self):
        self.stop = True
        self.is_llm_running = False

    def new_chat(self):
        self.stop = True
        self.is_llm_running = False
        self.chat_history_id.clear_widgets()
        self.messages = []

    def popup_smol135m_model(self):
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
                on_release=self.download_smol_135m_model
            ),
        ]
        self.show_text_dialog(
            "Downlaod the model file",
            f"You need to downlaod the file for the first time (~95MB)",
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
            self.is_downloading = False
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the onnx file: {e} 😞")
            Clock.schedule_once(lambda dt: self.show_toast_msg(f"Download failed for: {download_path}", is_error=True))
            self.is_downloading = False

    def download_model_file(self, model_url, download_path, instance=None):
        self.txt_dialog_closer(instance)
        filename = download_path.split("/")[-1]
        print(f"Starting the download for: {filename}")
        self.download_progress = self.root.ids.welcome_scr.ids.download_stat
        Thread(target=self.download_file, args=(model_url, download_path), daemon=True).start()

    def download_smol_135m_model(self, instance):
        url = self.llm_models[0]['url']
        model_name = self.llm_models[0]['name']
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
                    on_release=self.update_checker
                ),
            ]
            self.show_text_dialog(
                "Check for update",
                f"Your version: {__version__}",
                buttons
            )

    def llm_menu_callback(self, text_item):
        self.llm_menu.dismiss()
        self.selected_llm = text_item
        self.root.ids.chatbot_scr.ids.llm_menu.text = self.selected_llm
        self.init_onnx_sess(self.selected_llm)

    def token_menu_callback(self, text):
        self.token_menu.dismiss()
        self.token_count = int(text)
        self.root.ids.chatbot_scr.ids.token_menu.text = text

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

    def send_message(self, button_instance, chat_input_widget):
        if self.is_llm_running:
            self.show_toast_msg("Please wait for the current response", is_error=True)
            return
        user_message = chat_input_widget.text.strip()
        if user_message:
            user_message_add = f"{user_message}"
            self.messages.append(
                {
                    "role": "user",
                    "content": user_message
                }
            )
            self.add_usr_message(user_message_add)
            chat_input_widget.text = "" # blank the input
            self.tmp_txt = BotTmpResp()
            self.chat_history_id.add_widget(self.tmp_txt)
            msg_to_send = [{"role": "system", "content": "You are a helpful assistant."}]
            msg_to_send.extend(self.messages[-3:]) # taking last three messages only
            ollama_thread = Thread(target=self.chat_with_llm, args=(msg_to_send,), daemon=True)
            ollama_thread.start()
            self.is_llm_running = True
        else:
            self.show_toast_msg("Please type a message!", is_error=True)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_tensors="np"):
        prompt = ""  # no initials
        for msg in messages:
            role = msg["role"]
            content = msg["content"].strip()  # Strip for cleanliness
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant" or role == "model":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"

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
            self.eos_token_id = self.tokenizer.token_to_id("<|im_end|>")

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
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
            position_ids = np.tile(np.arange(0, input_ids.shape[-1]), (batch_size, 1))
            max_new_tokens = int(self.token_count)
            #generated_tokens = input_ids
            for i in range(max_new_tokens):
                logits, *present_key_values = self.decoder_session.run(None, dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **past_key_values,
                ))

                ## Update values for next generation loop
                #input_ids = np.argmax(logits[:, -1], axis=-1, keepdims=True)
                input_ids = self.sample_logits(logits[:, -1, :], temperature=0.7, top_p=0.9)
                attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids, dtype=np.int64)], axis=-1)
                position_ids = position_ids[:, -1:] + 1
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

                #generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
                if (input_ids == self.eos_token_id).all() or self.stop:
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