import json
import os
import subprocess

# 指定された設定ファイルの絶対パス
JSON_PATH = r"C:\tool\base_pyfile\base_pyfile\ai_models.json"

# グローバル変数にキャッシュを保持（2回目以降は Ollama を叩かない）
_cached_installed_models = None


def get_installed_ollama_models():
    """Ollamaに現在インストールされているモデル名の一覧を返す（初回のみ実行）"""
    global _cached_installed_models

    if _cached_installed_models is not None:
        return _cached_installed_models

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
        )

        installed_models = set()
        for line in result.stdout.strip().split("\n")[1:]:
            if line:
                model_name = line.split()[0]  # 行の最初の単語（モデル名）を取得
                installed_models.add(model_name)

        _cached_installed_models = installed_models
        print(
            "🔄 [Ollama] 実際にインストールされているモデルを取得し、キャッシュしました。"
        )
        return _cached_installed_models

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ Ollamaが起動していないか、コマンドが見つかりません。")
        return set()


def load_valid_models():
    """指定された JSON ファイルを読み込み、Ollamaに実在するモデルだけをフィルタリングして返す"""
    if not os.path.exists(JSON_PATH):
        print(f"❌ エラー: 指定されたパスに設定ファイルが見つかりません: {JSON_PATH}")
        return []

    # 1. 指定されたファイルをそのまま読み込む（上書きや作成は一切しません）
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 2. Ollamaのインストール済みリストを取得
    installed_models = get_installed_ollama_models()

    # 3. JSON内のモデルがOllamaにあるかチェック
    valid_models = []
    for model in config.get("models", []):
        # JSONに書かれている名前が、Ollamaに存在するか確認（クラウドモデル等はパスするよう考慮）
        if model["name"] in installed_models or ":cloud" in model["name"]:
            valid_models.append(model)
        else:
            print(
                f"❌ スキップ: {model['name']} はOllamaにインストールされていません。"
            )

    return valid_models

def format_models_for_router(available_models):
    """
    load_valid_models() から返ってきた利用可能なモデルリストを、
    ルーターAIが最も処理しやすい「識別番号付きのテキスト」に加工する関数
    """
    if not available_models:
        return "利用可能なモデルがありません。"

    lines = []
    
    # 1. 識別用の番号（1始まり）をループで振っていく
    for index, model in enumerate(available_models, start=1):
        name = model.get("name", "unknown")
        
        # 2. 判断に不要なdescriptionやroleは捨て、必要な情報だけを抽出
        # 万が一JSON側に項目がなかった場合（KeyError）を防ぐため、.get() で安全に取得
        best_for_list = model.get("best_for", [])
        avoid_for_list = model.get("avoid_for", [])
        
        # 3. ルーターが読みやすいように、リストを「、」区切りのシンプルな1行の文字列にする
        best_for_str = "、".join(best_for_list) if best_for_list else "特になし"
        avoid_for_str = "、".join(avoid_for_list) if avoid_for_list else "特になし"
        
        # 4. 余計な修飾文字（記号など）を減らし、AIが上から順にスキャンしやすい箇条書きにする
        model_text = (
            f"[{index}] {name}\n"
            f"  - 得意: {best_for_str}\n"
            f"  - 苦手: {avoid_for_str}"
        )
        lines.append(model_text)
    
    # すべてのモデルテキストを改行で連結
    return "\n".join(lines)



# ==========================================
# 🚀 実行処理
# ==========================================
if __name__ == "__main__":
    print("--- 1回目の実行（実際のollama listのチェックが走る） ---")
    available_models = load_valid_models()

    print("\n【✨ 現在利用可能なモデル一覧】")
    for m in available_models:
        print(f" 🟢 {m['name']}")

    print("\n--- 2回目の実行（キャッシュから一瞬で取得） ---")
    available_models_cached = load_valid_models()
    print(available_models_cached)


# ==========================================
# 🚀 動作確認（前回の関数からデータが渡ってきたと仮定）
# ==========================================
    # load_valid_models() から返ってくるデータのダミー（実際はJSONから自動で読み込まれます）
    mock_available_models = available_models_cached

    # 関数を実行して加工結果を取得
    ai_friendly_text = format_models_for_router(mock_available_models)
    
    print("【🤖 ルーターAIに渡すための、加工済みテキスト】\n")
    print(ai_friendly_text)
