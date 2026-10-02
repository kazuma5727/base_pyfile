"""テキストを指定した言語（既定は日本語）へ翻訳するモジュール。

`ai_corrector` と同じ設計です。クリップボードの中身をそのまま翻訳して書き戻す
用途（コピー → 貼り付けで翻訳結果が入っている）を想定しています。使用する
モデルは `ai_models.json` の `translation.tiers`（文字数のしきい値 → モデル）で
決めます。

失敗した場合（モデルが応答しない等）は、クリップボードを壊さないよう
**元のテキストをそのまま返します**。
"""

# --- 標準ライブラリのインポート ---
from logging import NullHandler, getLogger
from typing import Any, Dict, List, Optional

# --- 独自モジュールのインポート ---
from base_pyfile.ai_corrector import _clean_model_output, select_model_by_length
from base_pyfile.ai_router import (
    GENERATE_TIMEOUT,
    LIST_MODELS_TIMEOUT,
    generate,
    load_model_profiles,
)
from base_pyfile.log_setting import get_log_handler, make_logger

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

# よく使う言語コードと表示名の対応（プロンプトに埋め込む）
LANGUAGE_NAMES = {
    "ja": "日本語",
    "en": "英語",
    "zh": "中国語",
    "ko": "韓国語",
    "de": "ドイツ語",
    "fr": "フランス語",
    "es": "スペイン語",
    "pt": "ポルトガル語",
}

# 翻訳の既定言語（日本語へ翻訳）。-honyaku 起動はこれを利用します。
DEFAULT_TARGET_LANGUAGE = "ja"


def language_label(target_language: str) -> str:
    """言語コード（または言語名）を、プロンプト用の表示名へ変換します。

    Args:
        target_language (str): "ja" などの言語コード、または "日本語" のような表示名。

    Returns:
        str: 表示名。未知のコードはそのまま返します。
    """
    code = (target_language or "").strip().lower()
    return LANGUAGE_NAMES.get(code, target_language)


def build_translation_prompt(
    text: str, target_language: str = DEFAULT_TARGET_LANGUAGE
) -> str:
    """翻訳用のプロンプトを組み立てます。

    Args:
        text (str): 翻訳したいテキスト。
        target_language (str, optional): 翻訳先の言語。既定は "ja"（日本語）。

    Returns:
        str: モデルへ送るプロンプト。
    """
    label = language_label(target_language)
    return f"""あなたは正確な翻訳ツールです。以下のテキストを{label}に翻訳してください。

厳守事項:
- 意味やニュアンスをできるだけ忠実に保ってください。
- 説明・前置き・後書き・引用符・ローマ字読みを一切付けず、翻訳後のテキストだけを出力してください。
- 元のテキストが既に{label}の場合は、翻訳せずそのまま出力してください。
- コード、URL、コマンド、固有名詞は原則そのまま残してください。

【テキスト】
{text}

【{label}】"""


def translate_text(
    text: str,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    profiles: Optional[Dict[str, Any]] = None,
    installed: Optional[List[str]] = None,
    model: Optional[str] = None,
    timeout: int = GENERATE_TIMEOUT,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """テキストを指定言語へ翻訳し、結果を辞書で返します。

    Args:
        text (str): 翻訳したいテキスト。
        target_language (str, optional): 翻訳先の言語。既定は "ja"（日本語）。
        profiles (Optional[Dict[str, Any]], optional): モデル特性の辞書。
        installed (Optional[List[str]], optional): インストール済みモデルの一覧。
        model (Optional[str], optional): 使用するモデル名。省略した場合は
            文字数に応じて select_model_by_length() で自動選択します。
        timeout (int, optional): 生成のタイムアウト秒。デフォルトは GENERATE_TIMEOUT(120)。
        instruction (Optional[str], optional): プロンプトを丸ごと差し替えたい場合の指示文。
            指定した場合は末尾に【テキスト】を付けて使用します。

    Returns:
        Dict[str, Any]: 次のキーを持つ辞書。
            - "original": 元のテキスト
            - "translated": 翻訳後のテキスト（失敗時は元のテキスト）
            - "model": 実際に使用したモデル名
            - "changed": 内容が変化したかどうか
            - "elapsed_ms": 生成にかかった時間(ミリ秒)
            - "target_language": 実際に使用した翻訳先の言語
    """
    original = text
    if not text or not text.strip():
        return {
            "original": original,
            "translated": original,
            "model": None,
            "changed": False,
            "elapsed_ms": 0.0,
            "target_language": target_language,
        }

    if profiles is None:
        profiles = load_model_profiles()

    translation = profiles.get("translation", {}) or {}
    if len(text) < int(translation.get("min_chars", 1) or 1):
        return {
            "original": original,
            "translated": original,
            "model": None,
            "changed": False,
            "elapsed_ms": 0.0,
            "target_language": target_language,
        }

    if model is None:
        model = select_model_by_length(
            text,
            profiles=profiles,
            installed=installed,
            timeout=LIST_MODELS_TIMEOUT,
            section="translation",
        )

    if instruction:
        prompt = f"{instruction.strip()}\n\n【テキスト】\n{text}\n\n【翻訳後のテキスト】"
    else:
        prompt = build_translation_prompt(text, target_language=target_language)

    # 翻訳後のテキストが途中で切れないよう、文字数に応じて出力上限を確保する
    num_predict = min(max(64, len(text) * 2), 8192)

    translated, elapsed_ms = generate(
        prompt,
        model=model,
        options={"temperature": 0.0, "num_predict": num_predict},
        timeout=timeout,
    )

    if translated is None:
        # モデルが応答しなかった場合は元のテキストを維持する
        logger.warning(f"翻訳に失敗したため元のテキストを維持します (model={model})")
        translated = original
    else:
        translated = _clean_model_output(translated)

    if not translated:
        translated = original

    changed = translated != original
    if changed:
        logger.info(
            f"翻訳しました (model={model}, {len(original)}文字 -> {language_label(target_language)})"
        )
    else:
        logger.debug(f"内容は変化しませんでした (model={model})")

    return {
        "original": original,
        "translated": translated,
        "model": model,
        "changed": changed,
        "elapsed_ms": elapsed_ms,
        "target_language": target_language,
    }


def main() -> None:
    print("=== 翻訳ツール（日本語へ） ===")
    profiles = load_model_profiles()
    sample = "The quick brown fox jumps over the lazy dog."
    result = translate_text(sample, profiles=profiles)
    print(f"モデル: {result['model']}")
    print(f"元　　: {result['original']}")
    print(f"翻訳後: {result['translated']}")
    print(f"時間　: {result['elapsed_ms']:.1f} ms")


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))
    main()
