import os
import sys
import cv2
from kivy.app import App
from kivy.config import Config
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, BooleanProperty
from kivy.clock import Clock
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path

resource_add_path(r"C:\Windows\Fonts")
LabelBase.register(DEFAULT_FONT, "BIZ-UDMinchoM.ttc")
import configparser
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger, NullHandler

module_path = r"C:\tool\base_pyfile"
sys.path.append(module_path)
from log_setting import make_logger, get_log_handler
from path_manager import get_files, unique_path
from file_manager import read_text_file, write_file

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))


class ImageWidget(Widget):
    w, h = (1280, 480)
    on_Auto_mode = False
    width = "width"
    height = "height"
    Auto_mode = "Auto_mode"

    ini_file = f"{os.path.splitext(__file__)[0]}.ini"
    config = configparser.ConfigParser()
    config.read(ini_file)

    for section in config:
        w = config[section].get("width", w)
        h = config[section].get("height", h)
        on_Auto_mode = config[section].getboolean("Auto_mode", on_Auto_mode)

    Config.set("graphics", "width", w)
    Config.set("graphics", "height", h)
    Config.set("input", "mouse", "mouse, disable_multitouch")
    Config.set("modules", "inspector", "")
    on_off = BooleanProperty(on_Auto_mode)

    movie = StringProperty("")
    image = StringProperty("")

    def __init__(self, **kwargs):
        super(ImageWidget, self).__init__(**kwargs)

        # Auto_mode_checkのアクティブ状態をバインド
        self.ids.Auto_mode_check.bind(active=self.check_swich_temp)
        # 1つのワーカーしか許可しないスレッドプールを生成
        self.executor = ThreadPoolExecutor(max_workers=1)
        # ログにAuto_modeの状態を出力
        logger.debug(f"Auto_mode {self.on_off}")

        # Auto_modeがonの場合startメソッドを実行
        if self.on_off:
            logger.debug(f"Auto_mode {self.on_off}")
            self.start()

    def check_swich_temp(self, instance, valu):
        """
        Auto_modeのON/OFFを切り替えるためのメソッド
        """
        # Auto_modeの状態を変更
        self.on_off = valu
        # ログにAuto_modeの状態を出力
        logger.debug(f"{self.on_off}状態")

    def start(self):
        """
        画像を表示するためのメソッド
        """

        # 画像のパスを指定
        self.image = r"C:\tool\pyfile_folder\kivy_templates.jpg"
        # 画像を読み込み
        self.frame = cv2.imread(self.image)
        # 画像を表示するメソッド
        self.play()

    def Auto(self):
        """
        自動的に画像を表示するためのメソッド
        """

        print(self.on_off)
        # Auto_modeがonの場合
        if self.on_off:
            # Auto_modeをoffに変更
            self.on_off = False
            # Auto_btnのテキストを変更
            self.ids.Auto_btn.text = "Auto start mode OFF状態"
            # Clockのスケジュールを解除
            Clock.unschedule(self.update)
        # Auto_modeがoffの場合
        else:
            # Auto_modeをonに変更
            self.on_off = True
            # Auto_btnのテキストを変更
            self.ids.Auto_btn.text = "Auto start mode  ON状態"
            # startメソッドを実行
            self.start()

    def delete(self):
        """
        アプリを終了する際に、設定を保存するためのメソッド
        """
        from kivy.core.window import Window

        w, h = Window.size
        logger.debug(f"画面サイズ{w},{h}")

        for setting in ["KIVY_config", "KIVY_MODE"]:

            if "config" in setting:
                self.config[setting] = {self.width: w, self.height: h}

            if "MODE" in setting:
                self.config[setting] = {self.Auto_mode: self.on_off}

        with open(self.ini_file, "w") as configfile:
            # 指定したconfigファイルを書き込み
            self.config.write(configfile)

        sys.exit()

    def play(self):

        # self.cap = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update, 1 / 30)
        # Clock.schedule_once(self.update, 1 / 30)

        # Kivy Textureに変換

    def update(self, dt):
        # フレームを読み込み
        # ret, self.frame = self.cap.read()
        # Kivy Textureに変換
        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt="bgr")
        texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        # インスタンスのtextureを変更
        self.ids.camera.texture = texture


class KivyAPP(App):
    def __init__(self, **kwargs):
        super(KivyAPP, self).__init__(**kwargs)
        self.title = "タイトル"

    def build(self):
        return ImageWidget()


def kivy_app(kv_filepath=r"kivy_temp.kv"):

    # Builder.load_string("""
    # <App名>:
    # 内容
    # """)

    Builder.load_file(kv_filepath)
    KivyAPP().run()


if __name__ == "__main__":
    # logger = logger_create(debug := hdlr_create(10))
    kv_file = os.path.join(os.path.dirname(__file__),"kivy_temp.kv")

    kivy_app(kv_filepath=kv_file)
