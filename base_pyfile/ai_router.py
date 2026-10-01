"""渡された情報を、モデル特性JSONに基づいて最適なLLMへ振り分けるルーター。

モデル構成・振り分け設定はすべて `ai_models.json` 側で持ちます。Python側を
編集しなくても、JSONにモデルを追加・変更するだけで振り分け先を変えられます。

`ai_models.json` の "models" に書かれたモデルのうち、実際にインストール済みの
ものが振り分け先の候補になります。ルーター自身も候補に含め、判断の結果
そのルーターで足りる内容なら、ルーター自身に振って答えさせます。

振り分けの形式は2種類あります。
    - ラベル方式（既定）: ルーターには記号だけを答えさせる。記号 "A" は常に
      「わからない」を表す固定枠で、モデル候補は "B" 以降へ割り当てる。モデル名を
      出力させないため小型モデルでも崩れにくく、「わからない」の位置も動かない。
    - 名前方式: ルーターにモデル名を直接答えさせる（use_labels=False）。

Ollamaとの低レベルな通信もこのモジュールが担当します。
"""

# --- 標準ライブラリのインポート ---
import json
import time
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --- 外部ライブラリのインポート ---
import requests

# --- 独自モジュールのインポート ---
from base_pyfile.log_setting import get_log_handler, make_logger

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

# --- グローバル変数 ---
# Ollamaサーバーの接続先。モデル構成・振り分け設定は ai_models.json 側で管理する。
OLLAMA_BASE_URL = "http://localhost:11434"

# 「わからない」を表す固定ラベル。ラベル方式で常に同じ位置に固定する。
UNDECIDED_LABEL = "A"


def _model_profile_path() -> Path:
    """モデル特性JSON（ai_models.json）のパスを返します。"""
    return Path(__file__).with_name("ai_models.json")


# ============================================================
# 1. Ollamaとの低レベルな通信
# ============================================================
def list_installed_models(timeout: int = 5) -> List[str]:
    """Ollamaにインストール済みのモデル名一覧を取得します。

    Args:
        timeout (int, optional): 通信のタイムアウト秒。デフォルトは5。

    Returns:
        List[str]: モデル名（例: "tev1:0.8b"）のリスト。
            サーバーに接続できない場合は空のリストを返します。
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name", "") for model in models if model.get("name")]
    except Exception as e:
        logger.error(f"モデル一覧の取得に失敗しました: {e}")
        return []


def is_model_installed(
    model_name: str,
    models: Optional[Sequence[str]] = None,
    timeout: int = 5,
) -> bool:
    """指定されたモデルがインストール済みか判定します。

    Args:
        model_name (str): 判定するモデル名。
        models (Optional[Sequence[str]], optional): インストール済みモデルの一覧。
            省略した場合はOllamaへ問い合わせます。
        timeout (int, optional): 通信のタイムアウト秒。デフォルトは5。

    Returns:
        bool: インストール済みならTrue。モデル一覧を取得できない場合は、
            判断材料がないためTrueを返します。
    """
    if models is None:
        models = list_installed_models(timeout)

    # サーバーから一覧を得られない場合はフィルタしない
    if not models:
        return True

    if model_name in models:
        return True

    # タグを省略した名前でも一致とみなす（例: "qwen2.5" -> "qwen2.5:0.5b"）
    base_name = model_name.split(":")[0]
    return any(model.split(":")[0] == base_name for model in models)


def resolve_model(
    preferred: Optional[str] = None,
    fallback: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    timeout: int = 5,
) -> Optional[str]:
    """優先順にモデルを探し、実際に使うモデル名を1つ決定します。

    Args:
        preferred (Optional[str], optional): 最優先のモデル名。
        fallback (Optional[Sequence[str]], optional): 代替モデル名のリスト。
        models (Optional[Sequence[str]], optional): インストール済みモデルの一覧。
            省略した場合はOllamaへ問い合わせます。
        timeout (int, optional): 通信のタイムアウト秒。デフォルトは5。

    Returns:
        Optional[str]: 使用するモデル名。どれも見つからない場合は、
            インストール済みの先頭を返します。
    """
    if models is None:
        models = list_installed_models(timeout)

    for candidate in (preferred, *(fallback or [])):
        if candidate and is_model_installed(candidate, models):
            return candidate

    # 指定がどれも見つからない場合は、インストール済みの先頭を使う
    return models[0] if models else preferred


def generate(
    prompt: str,
    model: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 5,
) -> Tuple[Optional[str], float]:
    """Ollamaへプロンプトを送り、単発のテキスト生成を実行します。

    使用するモデルの選択（フォールバック含む）は呼び出し側で
    `resolve_model()` を使って行ってください。

    Args:
        prompt (str): モデルへ送るプロンプト。
        model (Optional[str], optional): 使用するモデル名。
            省略した場合は resolve_model() で自動選択します。
        options (Optional[Dict[str, Any]], optional): Ollamaの生成オプション
            （temperature, num_predict など）。省略した場合は空の辞書。
        timeout (int, optional): 通信のタイムアウト秒。デフォルトは5。

    Returns:
        Tuple[Optional[str], float]: 生成されたテキストと処理時間(ミリ秒)。
            失敗した場合は (None, 経過時間) を返します。
    """
    if model is None:
        model = resolve_model(timeout=timeout)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }

    start_time = time.time()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text, (time.time() - start_time) * 1000  # ミリ秒換算
    except Exception as e:
        logger.error(f"生成に失敗しました(model={model}): {e}")
        return None, (time.time() - start_time) * 1000


# ============================================================
# 2. モデル特性JSONの読み込み
# ============================================================
def load_model_profiles(profile_path: Optional[Path] = None) -> Dict[str, Any]:
    """モデル特性JSONを読み込みます。

    Args:
        profile_path (Optional[Path], optional): 読み込むJSONのパス。
            省略した場合は ai_models.json。

    Returns:
        Dict[str, Any]: "router" 設定と "models" 一覧を含む辞書。
            読み込みに失敗した場合は空の設定を返します。
    """
    path = Path(profile_path) if profile_path else _model_profile_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"モデル特性の読み込みに失敗しました: {path}, エラー: {e}")
        return {"router": {}, "models": []}


def get_available_profiles(
    profiles: Optional[Dict[str, Any]] = None,
    installed: Optional[Sequence[str]] = None,
    timeout: int = 5,
) -> List[Dict[str, Any]]:
    """振り分け先の候補となる（インストール済みの）モデルを抽出します。

    Args:
        profiles (Optional[Dict[str, Any]], optional): モデル特性の辞書。
            省略した場合は load_model_profiles() で読み込みます。
        installed (Optional[Sequence[str]], optional): インストール済みモデルの一覧。
            省略した場合はOllamaへ問い合わせます。
        timeout (int, optional): 通信のタイムアウト秒。デフォルトは5。

    Returns:
        List[Dict[str, Any]]: 利用可能なモデルの特性辞書のリスト。
    """
    if profiles is None:
        profiles = load_model_profiles()

    models = list(profiles.get("models", []))

    if installed is None:
        installed = list_installed_models(timeout)

    # サーバーから一覧を得られない場合はフィルタしない
    if not installed:
        return models

    return [m for m in models if is_model_installed(m.get("name", ""), installed)]


# ============================================================
# 3. モデル特性にもとづく振り分け判断
# ============================================================
def build_label_map(candidates: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """モデル候補をラベル（A, B, C, ...）へ割り当てます。

    ラベル "A" は常に「わからない」を表す固定枠とし、モデル候補の増減で
    わからない枠の位置が動かないようにします。モデル候補は "B" 以降へ
    ai_models.json に定義した順で割り当てます。

    Args:
        candidates (List[Dict[str, Any]]): 選択候補となるモデル特性のリスト。

    Returns:
        Dict[str, Optional[str]]: ラベルからモデル名への対応表。
            "A" の値は None（=わからない）です。
    """
    label_map: Dict[str, Optional[str]] = {UNDECIDED_LABEL: None}
    for index, model in enumerate(candidates):
        label_map[chr(ord("B") + index)] = model.get("name")
    return label_map


def _build_label_router_prompt(
    content: str,
    label_map: Dict[str, Optional[str]],
    candidates: List[Dict[str, Any]],
) -> str:
    """ラベル方式の振り分け用プロンプトを組み立てます。

    モデル名は出力させず、記号1文字だけを答えさせます。

    Args:
        content (str): 振り分け対象の入力内容。
        label_map (Dict[str, Optional[str]]): ラベルとモデル名の対応表。
        candidates (List[Dict[str, Any]]): 選択候補となるモデル特性のリスト。

    Returns:
        str: ルーター用のプロンプト。
    """
    profile_by_name = {model.get("name"): model for model in candidates}
    lines = [f"{UNDECIDED_LABEL}: わからない（適切なモデルを判断できない）"]
    for label, name in label_map.items():
        if name is None:
            continue
        profile = profile_by_name.get(name, {})
        description = profile.get("description", "")
        best_for = "、".join(profile.get("best_for", []))
        lines.append(f"{label}: {description} 得意: {best_for}")
    catalog = "\n".join(lines)

    return f"""あなたは入力内容を最適なAIモデルへ振り分けるルーターです。
以下の選択肢から、入力内容の処理に最も適したものを「記号1文字だけ」で答えてください。
モデル名や理由、説明、挨拶などは書かず、記号だけを出力してください。
どれが適切か判断できない場合は {UNDECIDED_LABEL} を選んでください。

【選択肢】
{catalog}

【入力内容】
{content}

記号："""


def _build_router_prompt(content: str, candidates: List[Dict[str, Any]]) -> str:
    """名前方式の振り分け用プロンプトを組み立てます。

    Args:
        content (str): 振り分け対象の入力内容。
        candidates (List[Dict[str, Any]]): 選択候補となるモデル特性のリスト。

    Returns:
        str: ルーター用のプロンプト。
    """
    lines = []
    for model in candidates:
        name = model.get("name", "")
        description = model.get("description", "")
        best_for = "、".join(model.get("best_for", []))
        lines.append(f"- {name}: {description} 得意: {best_for}")
    catalog = "\n".join(lines)

    return f"""あなたは入力内容を最適なAIモデルへ振り分けるルーターです。
以下のモデル一覧から、入力内容の処理に最も適したモデル名を「1つだけ」選んでください。
理由や説明、挨拶などは書かず、モデル名だけを出力してください。

【モデル一覧】
{catalog}

【入力内容】
{content}

モデル名："""


def _parse_label(
    text: Optional[str], label_map: Dict[str, Optional[str]]
) -> Optional[str]:
    """ルーターの出力から、有効なラベル1文字を特定します。

    Args:
        text (Optional[str]): ルーターが出力した文字列。
        label_map (Dict[str, Optional[str]]): 有効なラベルの集合。

    Returns:
        Optional[str]: 特定できたラベル。特定できない場合はNone。
    """
    if not text:
        return None
    for char in text.strip().upper():
        if char in label_map:
            return char
    return None


def _match_model_name(
    text: Optional[str], candidates: List[Dict[str, Any]]
) -> Optional[str]:
    """ルーターの出力から、候補に含まれるモデル名を特定します。

    Args:
        text (Optional[str]): ルーターが出力した文字列。
        candidates (List[Dict[str, Any]]): 選択候補となるモデル特性のリスト。

    Returns:
        Optional[str]: 一致したモデル名。特定できない場合はNone。
    """
    if not text:
        return None

    normalized = text.strip().lower()
    # 完全なモデル名（タグ付き）で一致するものを優先
    for model in candidates:
        name = model.get("name", "")
        if name and name.lower() in normalized:
            return name

    # タグを省略して出力された場合（例: "qwen2.5" -> "qwen2.5:0.5b"）に対応
    for model in candidates:
        name = model.get("name", "")
        base_name = name.split(":")[0].lower()
        if base_name and base_name in normalized:
            return name

    return None


def _find_router_model(profiles: Dict[str, Any]) -> Optional[str]:
    """role が "router" のモデル名を探します（メモの汎用情報から導出）。"""
    for model in profiles.get("models", []):
        if model.get("role") == "router":
            return model.get("name")
    return None


def _resolve_router_model(
    profiles: Dict[str, Any],
    router_model: Optional[str],
    installed: Optional[Sequence[str]],
) -> Optional[str]:
    """振り分け判断に使うルーターモデルを決定します。

    "router" 設定があれば優先し、無ければ role が "router" のモデルをメモから
    探します。特定の設定に依存せず、汎用のメモ情報だけでも動くようにするためです。
    """
    router = profiles.get("router", {})
    preferred = router_model or router.get("preferred") or _find_router_model(profiles)
    return resolve_model(
        preferred=preferred,
        fallback=router.get("fallback"),
        models=installed,
    )


def _ensure_installed(
    chosen: Optional[str],
    default_target: Optional[str],
    installed: Optional[Sequence[str]],
) -> Optional[str]:
    """選択モデルが実際に使えるか確認し、無ければ既定モデルへ切り替えます。"""
    if chosen and is_model_installed(chosen, installed):
        return chosen
    logger.warning(
        f"選択したモデル {chosen} が利用できないため、既定 {default_target} に切り替えます。"
    )
    return resolve_model(preferred=default_target, models=installed)


def route_to_model(
    content: str,
    profiles: Optional[Dict[str, Any]] = None,
    router_model: Optional[str] = None,
    installed: Optional[Sequence[str]] = None,
    use_labels: bool = True,
) -> Tuple[Optional[str], float]:
    """入力内容を分析し、処理を任せるモデル名を1つ決定します。

    Args:
        content (str): 振り分け対象の入力内容。
        profiles (Optional[Dict[str, Any]], optional): モデル特性の辞書。
            省略した場合は load_model_profiles() で読み込みます。
        router_model (Optional[str], optional): 振り分け判断に使うモデル名。
            省略した場合はJSONの "router.preferred" を使用します。
        installed (Optional[Sequence[str]], optional): インストール済みモデルの一覧。
            省略した場合はOllamaへ問い合わせます。
        use_labels (bool, optional): Trueならラベル方式、Falseなら名前方式で
            振り分けます。デフォルトはTrue。

    Returns:
        Tuple[Optional[str], float]: 選択されたモデル名と判断にかかった時間(ミリ秒)。
            利用可能なモデルが無い場合は (None, 0.0) を返します。
    """
    if profiles is None:
        profiles = load_model_profiles()

    candidates = get_available_profiles(profiles, installed)
    if not candidates:
        logger.warning("振り分け可能なモデルがありません。")
        return None, 0.0

    # 候補が1つだけならルーターを介さず即決
    if len(candidates) == 1:
        return candidates[0].get("name"), 0.0

    if use_labels:
        return _route_by_label(content, profiles, router_model, installed, candidates)
    return _route_by_name(content, profiles, router_model, installed, candidates)


def _route_by_label(
    content: str,
    profiles: Dict[str, Any],
    router_model: Optional[str],
    installed: Optional[Sequence[str]],
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[str], float]:
    """ラベル方式（A=わからない固定）でモデルを選びます。"""
    router = profiles.get("router", {})
    # 設定が無くても動くよう、既定は候補の先頭から自動で決める
    default_target = router.get("default_target") or candidates[0].get("name")
    safe_target = router.get("safe_target") or default_target

    label_map = build_label_map(candidates)
    resolved_router = _resolve_router_model(profiles, router_model, installed)
    prompt = _build_label_router_prompt(content, label_map, candidates)
    answer, elapsed_time = generate(
        prompt,
        model=resolved_router,
        options={"num_predict": 4, "temperature": 0.0},
    )

    label = _parse_label(answer, label_map)
    if label is None or label == UNDECIDED_LABEL:
        # わからない・解釈不能の場合は安全側に倒して安パイモデルを使う
        chosen = safe_target
        logger.info(
            f"ラベル={label or '解釈不能'} のため安パイモデル {chosen} を使用します。"
        )
    else:
        chosen = label_map.get(label)
        logger.debug(f"ラベル {label} -> モデル {chosen}")

    return _ensure_installed(chosen, default_target, installed), elapsed_time


def _route_by_name(
    content: str,
    profiles: Dict[str, Any],
    router_model: Optional[str],
    installed: Optional[Sequence[str]],
    candidates: List[Dict[str, Any]],
) -> Tuple[Optional[str], float]:
    """名前方式（モデル名を直接答えさせる）でモデルを選びます。"""
    router = profiles.get("router", {})
    default_target = router.get("default_target") or candidates[0].get("name")

    resolved_router = _resolve_router_model(profiles, router_model, installed)
    prompt = _build_router_prompt(content, candidates)
    answer, elapsed_time = generate(
        prompt,
        model=resolved_router,
        options={"num_predict": 24, "temperature": 0.0},
    )

    chosen = _match_model_name(answer, candidates)
    if chosen is None:
        logger.warning(
            f"振り分け結果を解釈できませんでした: {answer!r} -> 既定の {default_target} を使用します。"
        )
        chosen = default_target

    return _ensure_installed(chosen, default_target, installed), elapsed_time


def dispatch(
    content: str,
    profiles: Optional[Dict[str, Any]] = None,
    router_model: Optional[str] = None,
    installed: Optional[Sequence[str]] = None,
    use_labels: bool = True,
) -> Dict[str, Any]:
    """入力内容を最適なモデルへ振り分け、そのモデルで処理を実行します。

    Args:
        content (str): 処理させたい入力内容。
        profiles (Optional[Dict[str, Any]], optional): モデル特性の辞書。
        router_model (Optional[str], optional): 振り分け判断に使うモデル名。
        installed (Optional[Sequence[str]], optional): インストール済みモデルの一覧。
        use_labels (bool, optional): ラベル方式で振り分けるか。デフォルトはTrue。

    Returns:
        Dict[str, Any]: 次のキーを持つ結果の辞書。
            - "model": 実際に使用したモデル名
            - "answer": モデルの応答（失敗時はNone）
            - "route_ms": 振り分けにかかった時間(ミリ秒)
            - "generate_ms": 生成にかかった時間(ミリ秒)
            - "total_ms": 合計時間(ミリ秒)
    """
    chosen, route_time = route_to_model(
        content,
        profiles=profiles,
        router_model=router_model,
        installed=installed,
        use_labels=use_labels,
    )

    if chosen is None:
        return {
            "model": None,
            "answer": None,
            "route_ms": route_time,
            "generate_ms": 0.0,
            "total_ms": route_time,
        }

    answer, generate_time = generate(content, model=chosen)
    return {
        "model": chosen,
        "answer": answer,
        "route_ms": route_time,
        "generate_ms": generate_time,
        "total_ms": route_time + generate_time,
    }


def main():
    print("=== Ollama 振り分けルーターのテストを開始 ===")

    profiles = load_model_profiles()
    installed = list_installed_models()
    available = get_available_profiles(profiles, installed)

    print("\n[振り分け先（使ってもらうモデル）]")
    for model in available:
        print(f"  - {model.get('name')} ({model.get('role')}): {model.get('description')}")

    # ラベル方式の割り当てを表示（A は常に「わからない」固定）
    print("\n[ラベル割り当て]")
    for label, name in build_label_map(available).items():
        print(f"  {label}: {name or 'わからない（安パイモデル）'}")

    # 振り分け対象のサンプル
    samples = [
        "フィボナッチ数列を返すPython関数を書いて。",
        "次の文章を一文で要約して。",
        "3人の誕生日が同じになる確率を順を追って説明して。",
        "この色は赤か青か、どちらかに答えて。",
    ]

    for content in samples:
        chosen, route_time = route_to_model(
            content, profiles=profiles, installed=installed
        )
        print("\n--------------------------------------------------")
        print(f"入力: {content}")
        print(f"👉 振り分け先: {chosen}  (判断: {route_time:.1f} ミリ秒)")

    # 実際に振り分けて実行する例（コストが高いため1件のみ）
    print("\n=== dispatch() の実行例 ===")
    result = dispatch(samples[3], profiles=profiles, installed=installed)
    print(f"モデル: {result['model']}")
    print(f"応答 : {result['answer']}")
    print(f"時間 : {result['total_ms']:.1f} ミリ秒")


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    main()
