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
from myrst import MyRstDocument
from screens.chatbot_screen import TempSpinWait, ChatbotScreen

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

## debug

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
            },
            "Try Other Apps": {
                "icon": "google-play",
                "action": "web",
                "url": "https://daslearning.in/apps/",
            },
        }
        return Builder.load_file(kv_file_path)

    def on_start(self):
        self.llm_models = ["smollm2-135m"]
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
        self.init_onnx_sess()
        print("Initialisation is successful")

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

    def add_bot_message(self, msg_to_add):
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
            self.tmp_txt = MDLabel(
                size_hint_y=None,
                #markup=True,
                halign='left',
                valign='top',
                padding=[dp(10), dp(10)],
                font_style="Subtitle1",
                allow_selection = True,
                allow_copy = True,
                text = "",
            )
            self.tmp_txt.bind(texture_size=self.tmp_txt.setter('size'))
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
        print(f"logits dtype: {logits.dtype} min: {np.min(logits)} max: {np.max(logits)}")
        print(f"logits shape: {logits.shape}")
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
                print("Next token id:", int(input_ids[0][0]))
                attention_mask = np.concatenate([attention_mask, np.ones_like(input_ids, dtype=np.int64)], axis=-1)
                position_ids = position_ids[:, -1:] + 1
                for j, key in enumerate(past_key_values):
                    past_key_values[key] = present_key_values[j]

                #generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
                if (input_ids == self.eos_token_id).any():
                    break

                ## (Optional) Streaming (use tokenizer.decode)
                txt_update = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                if txt_update:
                    print(f"Decoded: {txt_update}")
                    final_txt += str(txt_update)
                    Clock.schedule_once(lambda dt: self.update_text_stream(txt_update))
            # final result
            final_result["content"] = final_txt
            final_result["role"] = "assistant"
            print(f"Final text: {final_txt}")
        except Exception as e:
            print(f"Chat error: {e}")
            final_result["content"] = f"**Error** with LLM: {e}"
            final_result["role"] = "error"
        Clock.schedule_once(lambda dt: self.final_llm_result(final_result))

    def update_text_stream(self, txt_update):
        if self.tmp_txt:
            self.tmp_txt.text = self.tmp_txt.text + txt_update

    def final_llm_result(self, llm_resp):
        if llm_resp["role"] == "assistant":
            self.messages.append(llm_resp)
        self.is_llm_running = False
        txt = llm_resp["content"]
        self.chat_history_id.remove_widget(self.tmp_txt)
        self.add_bot_message(msg_to_add=txt)

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