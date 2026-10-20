"""クリップボードの画像からQRコードを見つけてURLを開く、Windows専用の常駐ツール。

使い方:
    python -m pip install Pillow pyzbar
    python QRreader.py

「Windows + Shift + S」でQRコードをキャプチャすると、自動でURLが開きます。
停止は Ctrl+C。base_pyfile パッケージには含まれない独立スクリプトです。
"""

import ctypes
import sys
import webbrowser
from ctypes import wintypes
from urllib.parse import urlparse

if sys.platform != "win32":
    raise SystemExit("QRreader.py は Windows 専用です。")

try:
    from PIL import Image, ImageGrab
    from pyzbar.pyzbar import decode
except ImportError as exc:
    raise SystemExit(
        "Pillow と pyzbar が必要です: python -m pip install Pillow pyzbar"
    ) from exc

# Windows APIの最低限必要な関数を直接定義（環境に左右されない汎用ポインタ型を使用）
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WM_CLIPBOARDUPDATE = 0x031D
WM_DESTROY = 0x0002

is_processing = False


def is_http_url(value: str) -> bool:
    """http:// または https:// のURLだけを許可する。"""
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def scan_qr_from_clipboard():
    """クリップボードの画像を解析してQRコードを探す"""
    global is_processing
    if is_processing:
        return

    try:
        is_processing = True
        img = ImageGrab.grabclipboard()

        if isinstance(img, Image.Image):
            if img.width > 2000 or img.height > 2000:
                img.thumbnail((1500, 1500))

            detected_barcodes = decode(img)
            if detected_barcodes:
                for barcode in detected_barcodes:
                    url = barcode.data.decode("utf-8")
                    if not is_http_url(url):
                        print(f"\n⚠️ URL以外のQRコードのため開きません: {url}")
                        continue
                    print(f"\n🎉 QRコードを発見しました: {url}")
                    webbrowser.open(url)
                    break
    except Exception as e:
        print(f"解析エラー: {e}")
    finally:
        is_processing = False


def wnd_proc(hwnd, msg, wparam, lparam):
    """Windowsからの通知を受け取る窓口"""
    if msg == WM_CLIPBOARDUPDATE:
        scan_qr_from_clipboard()
        return 0
    elif msg == WM_DESTROY:
        user32.RemoveClipboardFormatListener(hwnd)
        user32.PostQuitMessage(0)
        return 0
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


def start_watcher():
    print("🟢 システム起動完了。常時監視中...")
    print("「Windows + Shift + S」でQRコードをスクショすると、自動でURLが開きます。")
    print("（終了するには、この画面で Ctrl + C を押してください）")

    # 特殊な型を一切使わず、すべて汎用ポインタ（c_void_p）と整数型だけでウィンドウクラスを定義
    WNDPROC = ctypes.WINFUNCTYPE(
        ctypes.c_int64, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", ctypes.c_void_p),
            ("hIcon", ctypes.c_void_p),
            ("hCursor", ctypes.c_void_p),
            ("hbrBackground", ctypes.c_void_p),
            ("lpszMenuName", ctypes.c_wchar_p),
            ("lpszClassName", ctypes.c_wchar_p),
        ]

    wnd_proc_ptr = WNDPROC(wnd_proc)
    hInstance = kernel32.GetModuleHandleW(None)

    wc = WNDCLASSW()
    wc.lpfnWndProc = wnd_proc_ptr
    wc.hInstance = hInstance
    wc.lpszClassName = "QRClipboardWatcherClassUnique"

    if not user32.RegisterClassW(ctypes.byref(wc)):
        print("❌ ウィンドウクラスの登録に失敗しました。")
        return

    # メッセージ受信用の見えない隠しウィンドウを作成
    hwnd = user32.CreateWindowExW(
        0, wc.lpszClassName, "QR Watcher", 0, 0, 0, 0, 0, None, None, hInstance, None
    )
    if not hwnd:
        print("❌ 監視用ウィンドウの作成に失敗しました。")
        return

    # WindowsのOSに「クリップボードが変わったらこのウィンドウに通知して」と登録
    # OS側で変更があった時だけプログラムが叩かれるため、CPU負荷は0%です
    if not user32.AddClipboardFormatListener(hwnd):
        print("❌ OSへの監視登録に失敗しました。")
        return

    # Windowsからの通知待ちループ（イベントが来るまで完全にスリープします）
    msg = wintypes.MSG()
    try:
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    except KeyboardInterrupt:
        print("\n🛑 監視を終了しました。")
        user32.DestroyWindow(hwnd)


if __name__ == "__main__":
    start_watcher()
