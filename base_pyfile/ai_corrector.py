"""テキストの誤字・脱字を、文字数に応じて選んだモデルで校正するモジュール。

クリップボードの中身をそのまま校正して書き戻す用途（コピー → 貼り付けで直っている）
を想定しています。振り分けは `ai_router` のラベル分類ではなく、`ai_models.json` の
`correction.tiers`（文字数のしきい値 → モデル）で決めます。

失敗した場合（モデルが応答しない等）は、クリップボードを壊さないよう
**元のテキストをそのまま返します**。
"""

# --- 標準ライブラリのインポート ---
from logging import NullHandler, getLogger
from typing import Any, Dict, List, Optional

# --- 独自モジュールのインポート ---
from base_pyfile.ai_router import (
    GENERATE_TIMEOUT,
    LIST_MODELS_TIMEOUT,
    generate,
    load_model_profiles,
    resolve_model,
)
from base_pyfile.log_setting import get_log_handler, make_logger

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


DEFAULT_INSTRUCTION = """あなたは日本語と英語の校正ツールです。以下のテキストに含まれる
誤字・脱字・打ち間違い・明らかな文法的誤りだけを修正してください。

厳守事項:
- 意味や内容は変えないでください。
- 表記ゆれの統一や、過度な言い換えはしないでください。
- コード、URL、コマンド、固有名詞は変更しないでください。
- 説明・前置き・後書き・引用符を一切付けず、修正後のテキストだけを出力してください。
- 修正する箇所が無い場合は、元のテキストをそのまま出力してください。"""


def build_correction_prompt(text: str, instruction: Optional[str] = None) -> str:
    """校正用のプロンプトを組み立てます。

    Args:
        text (str): 校正したいテキスト。
        instruction (Optional[str], optional): 校正方針の指示。省略した場合は既定の指示。

    Returns:
        str: モデルへ送るプロンプト。
    """
    return f"{(instruction or DEFAULT_INSTRUCTION).strip()}\n\n【テキスト】\n{text}\n\n【修正後のテキスト】"


def select_model_by_length(
    text: str,
    profiles: Optional[Dict[str, Any]] = None,
    installed: Optional[List[str]] = None,
    timeout: int = LIST_MODELS_TIMEOUT,
    section: str = "correction",
) -> Optional[str]:
    """文字数に応じて処理に使うモデル名を1つ選びます。

    `ai_models.json` の指定セクション（既定は `correction`）の `tiers` を
    `max_chars` 昇順に見て、文字数が収まる最初のモデルを選びます。
    未インストールだった場合は同セクションの `fallback` →
    `router.default_target` の順に切り替えます。

    Args:
        text (str): 処理対象のテキスト。
        profiles (Optional[Dict[str, Any]], optional): モデル特性の辞書。
            省略した場合は load_model_profiles() で読み込みます。
        installed (Optional[List[str]], optional): インストール済みモデルの一覧。
            省略した場合は Ollama へ問い合わせます。
        timeout (int, optional): 一覧取得のタイムアウト秒。
        section (str, optional): 参照する設定セクション名。既定は "correction"。
            翻訳では `ai_translator` から "translation" が渡されます。

    Returns:
        Optional[str]: 選択されたモデル名。決められない場合は None。
    """
    if profiles is None:
        profiles = load_model_profiles()

    correction = profiles.get(section, {}) or {}
    tiers = [t for t in correction.get("tiers", []) if isinstance(t, dict)]
    # max_chars が None のものは最後尾に回して昇順ソート
    tiers.sort(key=lambda t: (t.get("max_chars") is None, t.get("max_chars") or 0))

    length = len(text)
    preferred: Optional[str] = None
    for tier in tiers:
        max_chars = tier.get("max_chars")
        if max_chars is None or length <= max_chars:
            preferred = tier.get("model")
            break

    if not preferred:
        preferred = (profiles.get("router", {}) or {}).get("default_target")

    fallback = correction.get("fallback") or []
    return resolve_model(
        preferred=preferred, fallback=fallback, models=installed, timeout=timeout
    )


def _clean_model_output(text: str) -> str:
    """モデルが付けてしまった余計な引用符やコードフェンスを除去します。"""
    cleaned = text.strip()

    # ```...``` で囲まれていたら外す
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # 全体を囲む引用符（"..." や “...” など）を外す
    pairs = [('"', '"'), ("'", "'"), ("「", "」"), ("“", "”"), ("『", "』")]
    for start, end in pairs:
        if len(cleaned) >= 2 and cleaned.startswith(start) and cleaned.endswith(end):
            cleaned = cleaned[1:-1].strip()

    return cleaned


def correct_text(
    text: str,
    profiles: Optional[Dict[str, Any]] = None,
    installed: Optional[List[str]] = None,
    model: Optional[str] = None,
    timeout: int = GENERATE_TIMEOUT,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """テキストの誤字・脱字を校正し、結果を辞書で返します。

    Args:
        text (str): 校正したいテキスト。
        profiles (Optional[Dict[str, Any]], optional): モデル特性の辞書。
        installed (Optional[List[str]], optional): インストール済みモデルの一覧。
        model (Optional[str], optional): 使用するモデル名。省略した場合は
            文字数に応じて select_model_by_length() で自動選択します。
        timeout (int, optional): 生成のタイムアウト秒。デフォルトは GENERATE_TIMEOUT(120)。
        instruction (Optional[str], optional): 校正方針を差し替えたい場合の指示文。

    Returns:
        Dict[str, Any]: 次のキーを持つ辞書。
            - "original": 元のテキスト
            - "corrected": 校正後のテキスト（失敗時は元のテキスト）
            - "model": 実際に使用したモデル名
            - "changed": 内容が変化したかどうか
            - "elapsed_ms": 生成にかかった時間(ミリ秒)
    """
    original = text
    if not text or not text.strip():
        return {
            "original": original,
            "corrected": original,
            "model": None,
            "changed": False,
            "elapsed_ms": 0.0,
        }

    if profiles is None:
        profiles = load_model_profiles()

    correction = profiles.get("correction", {}) or {}
    if len(text) < int(correction.get("min_chars", 1) or 1):
        return {
            "original": original,
            "corrected": original,
            "model": None,
            "changed": False,
            "elapsed_ms": 0.0,
        }

    if model is None:
        model = select_model_by_length(text, profiles=profiles, installed=installed)

    prompt = build_correction_prompt(text, instruction=instruction)
    # 校正後のテキストが途中で切れないよう、文字数に応じて出力上限を確保する
    num_predict = min(max(64, len(text) * 2), 8192)

    corrected, elapsed_ms = generate(
        prompt,
        model=model,
        options={"temperature": 0.0, "num_predict": num_predict},
        timeout=timeout,
    )

    if corrected is None:
        # モデルが応答しなかった場合は元のテキストを維持する
        logger.warning(f"校正に失敗したため元のテキストを維持します (model={model})")
        corrected = original
    else:
        corrected = _clean_model_output(corrected)

    if not corrected:
        corrected = original

    changed = corrected != original
    if changed:
        logger.info(f"校正しました (model={model}, {len(original)}文字)")
    else:
        logger.debug(f"修正箇所はありませんでした (model={model})")

    return {
        "original": original,
        "corrected": corrected,
        "model": model,
        "changed": changed,
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    print("=== 誤字脱字 校正ツール ===")
    profiles = load_model_profiles()
    sample = "これはてすとです。誤字脱字をなおしてくれると助かります。"
    result = correct_text(sample, profiles=profiles)
    print(f"モデル: {result['model']}")
    print(f"元　　: {result['original']}")
    print(f"校正後: {result['corrected']}")
    print(f"時間　: {result['elapsed_ms']:.1f} ms")


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))
    main()
