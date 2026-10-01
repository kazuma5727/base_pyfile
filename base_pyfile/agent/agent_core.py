"""指示を受けて、Pythonコードを書きながら自立的に作業を進めるローカルLLMエージェント。

このモジュールは、Ollamaへ自動接続し、`ai_models.json` のモデルメモから
エージェント本体に使うモデルを選びます。指示を渡すと、モデルにコードを書かせ、
それを実行して結果を次の判断材料として渡す、という流れを繰り返します。
最終的に「指示を満たすプログラム」を書き上げて答えを返そうとします。

生成されたコードの実行方法は2通りあります。
    - "subprocess"（既定）: 作業フォルダへ .py として保存し、別プロセスで実行する。
      生成物が残り、暴走しても本体へ影響しにくい。
    - "exec": 本体プロセス内で実行する。base_pyfile の関数を名前空間へ直接渡せる。

モデルや実行方法・作業フォルダなどの設定は `ai_models.json` の "agent" ブロックに
書けます。設定が無い場合でも、メモ（models）だけから妥当な既定を導いて動きます。
"""

# --- 標準ライブラリのインポート ---
import contextlib
import inspect
import io
import re
import subprocess
import sys
import time
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# --- 独自モジュールのインポート ---
from base_pyfile.ai_router import (
    generate,
    is_model_installed,
    list_installed_models,
    load_model_profiles,
    resolve_model,
)
from base_pyfile.file_manager import write_file
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import make_directory

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def _agent_dir() -> Path:
    """エージェント自身が置かれているフォルダ（base_pyfile/agent）を返します。"""
    return Path(__file__).parent


def _project_root() -> Path:
    """base_pyfile を import できるリポジトリのルートを返します。"""
    return Path(__file__).resolve().parents[2]


def _resolve_work_dir(work_dir: Optional[Union[str, Path]]) -> Path:
    """作業フォルダを解決し、無ければ作成します。

    相対パスはエージェントのフォルダ基準で解決します。

    Args:
        work_dir (Optional[Union[str, Path]]): 作業フォルダ。Noneの場合は
            エージェントフォルダ内の "workspace"。

    Returns:
        Path: 作成済みの作業フォルダのパス。
    """
    if work_dir is None:
        path = _agent_dir() / "workspace"
    else:
        path = Path(work_dir)
        if not path.is_absolute():
            path = _agent_dir() / path
    make_directory(path)
    return path


# ============================================================
# 1. Ollamaへの自動接続とモデル選択
# ============================================================
def wait_for_ollama(retries: int = 3, interval: float = 1.0, timeout: int = 5) -> bool:
    """Ollamaサーバーへ接続できるまで、少し待ちながら再試行します。

    Args:
        retries (int, optional): 接続を試みる回数。デフォルトは3。
        interval (float, optional): 再試行までの待ち時間(秒)。デフォルトは1.0。
        timeout (int, optional): 1回あたりの通信タイムアウト秒。デフォルトは5。

    Returns:
        bool: 接続できた場合はTrue。最後まで接続できなければFalse。
    """
    for attempt in range(1, retries + 1):
        if list_installed_models(timeout):
            logger.info("Ollamaへの接続を確認しました。")
            return True
        logger.warning(
            f"Ollamaへ接続できません（{attempt}/{retries}）。{interval}秒後に再試行します。"
        )
        time.sleep(interval)

    logger.error("Ollamaへ接続できませんでした。サーバーが起動しているか確認してください。")
    return False


def load_agent_settings(profiles: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`ai_models.json` の "agent" 設定ブロックを読み込みます。

    Args:
        profiles (Optional[Dict[str, Any]], optional): 読み込み済みの設定辞書。
            省略した場合は load_model_profiles() で読み込みます。

    Returns:
        Dict[str, Any]: "agent" ブロックの内容。無い場合は空の辞書。
    """
    if profiles is None:
        profiles = load_model_profiles()
    return profiles.get("agent", {}) or {}


def resolve_agent_model(
    settings: Optional[Dict[str, Any]] = None,
    installed: Optional[List[str]] = None,
    timeout: int = 5,
) -> Optional[str]:
    """エージェント本体に使うモデルを1つ決定します。

    "agent" 設定の `model` を優先し、無ければメモの `role` から
    コーディング向きのモデルを導いて選びます。未インストールのモデルは
    `fallback` やインストール済み一覧から代替します。

    Args:
        settings (Optional[Dict[str, Any]], optional): "agent" 設定ブロック。
            省略した場合は load_agent_settings() で読み込みます。
        installed (Optional[List[str]], optional): インストール済みモデルの一覧。
            省略した場合はOllamaへ問い合わせます。
        timeout (int, optional): 通信のタイムアウト秒。デフォルトは5。

    Returns:
        Optional[str]: 使用するモデル名。決定できなければNone。
    """
    if settings is None:
        settings = load_agent_settings()

    if installed is None:
        installed = list_installed_models(timeout)

    preferred = settings.get("model")
    fallback = settings.get("fallback") or []

    if preferred:
        return resolve_model(preferred=preferred, fallback=fallback, models=installed)

    # 設定が無い場合はメモの role からコーディング向きのモデルを導く
    profiles = load_model_profiles()
    for role in ("code", "reasoning", "light", "router"):
        for model in profiles.get("models", []):
            name = model.get("name", "")
            if model.get("role") == role and is_model_installed(name, installed):
                return name

    return resolve_model(preferred=None, fallback=fallback, models=installed)


# ============================================================
# 2. プロンプトの組み立て
# ============================================================
def available_functions() -> str:
    """base_pyfile が公開している関数を、引数付きの一覧テキストにして返します。

    Returns:
        str: 「- 関数名(引数)」を1行ずつ並べたテキスト。
    """
    import base_pyfile

    lines: List[str] = []
    for name in getattr(base_pyfile, "__all__", []):
        obj = getattr(base_pyfile, name, None)
        if not callable(obj):
            continue
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            signature = "(...)"
        lines.append(f"- {name}{signature}")
    return "\n".join(lines)


def _format_history(history: List[Dict[str, Any]]) -> str:
    """これまでの作業履歴を、モデルへ渡すテキストに整形します。"""
    if not history:
        return "（まだ作業していません）"

    lines: List[str] = []
    for record in history:
        result = record.get("result", {})
        lines.append(f"--- ステップ{record['step']} のコード ---")
        lines.append(record.get("code", ""))
        lines.append(f"--- ステップ{record['step']} の実行結果 ---")
        if result.get("output"):
            lines.append(result["output"])
        if result.get("error"):
            lines.append(f"エラー: {result['error']}")
        if not result.get("output") and not result.get("error"):
            lines.append("（出力なし。結果は print() で出力してください）")
    return "\n".join(lines)


def build_prompt(
    instruction: str,
    history: List[Dict[str, Any]],
    functions_text: str,
    work_dir: Path,
) -> str:
    """エージェントの1ステップ分のプロンプトを組み立てます。

    Args:
        instruction (str): ユーザーからの指示。
        history (List[Dict[str, Any]]): これまでのコードと実行結果の履歴。
        functions_text (str): 利用可能な base_pyfile 関数の一覧。
        work_dir (Path): 作業フォルダのパス。

    Returns:
        str: モデルへ送るプロンプト。
    """
    return f"""あなたは、指示を達成するためにPythonコードを書きながら作業を進めるエージェントです。
必ず次のどちらか一方だけを出力してください。

1. まだ作業が必要な場合: ```python で囲んだコードブロックを1つだけ出力する。
   そのコードは自動で実行され、標準出力とエラーが結果として次の入力に渡されます。
   結果を確認できるよう、必ず print() で出力してください。
2. 指示を達成できた場合: 「FINAL:」に続けて、最終的な答えや成果を書く。

説明や挨拶は最小限にし、コードブロックか FINAL のどちらかを必ず含めてください。

【利用可能な関数】base_pyfile から次の関数をそのまま使えます（別プロセス実行時は `import base_pyfile` してください）
{functions_text}

【作業フォルダ】ここで作成したファイルはこのフォルダに保存されます
{work_dir}

【指示】
{instruction}

【これまでの作業】
{_format_history(history)}

【次の行動】"""


# ============================================================
# 3. モデル出力の解析
# ============================================================
def _extract_code(text: Optional[str]) -> Optional[str]:
    """モデルの出力からPythonコードを取り出します。

    コードフェンス（```python ... ```）を優先し、無い場合はコードらしい
    行（import / def / print などで始まる行）があれば全体をコードとみなします。

    Args:
        text (Optional[str]): モデルの出力。

    Returns:
        Optional[str]: 取り出したコード。コードが無ければNone。
    """
    if not text:
        return None

    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip() or None

    # フェンス無しでもコードらしければコードとして扱う
    keyword = re.compile(r"^\s*(import|from|def|class|for|while|if|print|with|try)\b")
    if any(keyword.match(line) for line in text.splitlines()):
        return text.strip()
    return None


def _extract_final(text: Optional[str]) -> Optional[str]:
    """モデルの出力から最終回答（FINAL: 以降）を取り出します。

    Args:
        text (Optional[str]): モデルの出力。

    Returns:
        Optional[str]: 最終回答。見つからなければNone。
    """
    if not text:
        return None

    match = re.search(r"(?:FINAL|最終回答|完了)\s*[:：]\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ============================================================
# 4. 生成コードの実行
# ============================================================
def _execute_subprocess(
    code: str, work_dir: Path, step: int, timeout: int
) -> Dict[str, Any]:
    """コードを .py として保存し、別プロセスで実行します。"""
    root = _project_root()
    header = (
        "import sys as _sys\n"
        f"if {str(root)!r} not in _sys.path:\n"
        f"    _sys.path.insert(0, {str(root)!r})\n"
        "# --- ここからエージェントが生成したコード ---\n"
    )
    script_path = work_dir / f"step_{step}.py"
    write_file(script_path, header + code + "\n", extension=".py", backup=False)

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "file": str(script_path),
            "output": "",
            "error": f"{timeout}秒でタイムアウトしました。",
        }
    except Exception as e:
        return {
            "ok": False,
            "file": str(script_path),
            "output": "",
            "error": f"{type(e).__name__}: {e}",
        }

    return {
        "ok": completed.returncode == 0,
        "file": str(script_path),
        "output": (completed.stdout or "").strip(),
        "error": (completed.stderr or "").strip() or None,
    }


def _execute_in_process(
    code: str, work_dir: Path, step: int, timeout: int
) -> Dict[str, Any]:
    """base_pyfile の関数を名前空間へ渡し、本体プロセス内でコードを実行します。"""
    import base_pyfile

    namespace: Dict[str, Any] = {"__name__": "__agent_exec__"}
    for name in getattr(base_pyfile, "__all__", []):
        namespace[name] = getattr(base_pyfile, name, None)

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            exec(compile(code, f"<agent step {step}>", "exec"), namespace)
        return {
            "ok": True,
            "file": None,
            "output": buffer.getvalue().strip(),
            "error": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "file": None,
            "output": buffer.getvalue().strip(),
            "error": f"{type(e).__name__}: {e}",
        }


def execute_code(
    code: str,
    mode: str = "subprocess",
    work_dir: Optional[Union[str, Path]] = None,
    step: int = 1,
    timeout: int = 60,
) -> Dict[str, Any]:
    """生成されたコードを実行し、その結果を辞書で返します。

    Args:
        code (str): 実行するPythonコード。
        mode (str, optional): "subprocess"（保存して別プロセス実行、既定）または
            "exec"（本体プロセス内で実行）。
        work_dir (Optional[Union[str, Path]], optional): 作業フォルダ。
            省略した場合はエージェントフォルダ内の "workspace"。
        step (int, optional): ステップ番号。保存するファイル名に使います。デフォルトは1。
        timeout (int, optional): 実行のタイムアウト秒。デフォルトは60。

    Returns:
        Dict[str, Any]: 次のキーを持つ辞書。
            - "ok": 成功したか
            - "file": 保存したファイルのパス（exec時はNone）
            - "output": 標準出力
            - "error": エラー内容（無ければNone）
    """
    directory = _resolve_work_dir(work_dir)
    if mode == "exec":
        logger.warning("exec方式ではタイムアウトを適用できません。")
        return _execute_in_process(code, directory, step, timeout)
    return _execute_subprocess(code, directory, step, timeout)


# ============================================================
# 5. エージェント本体のループ
# ============================================================
def run_agent(
    instruction: str,
    model: Optional[str] = None,
    max_steps: Optional[int] = None,
    execute: Optional[str] = None,
    work_dir: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """指示を達成するまで、コードを書いて実行する流れを繰り返します。

    引数を省略した項目は、`ai_models.json` の "agent" 設定、それも無ければ
    組み込みの既定値（max_steps=8, execute="subprocess", timeout=60）を使います。

    Args:
        instruction (str): 達成したい指示。
        model (Optional[str], optional): 使用するモデル名。省略時は自動選択。
        max_steps (Optional[int], optional): コード生成〜実行を繰り返す上限回数。
        execute (Optional[str], optional): 実行方式（"subprocess" / "exec"）。
        work_dir (Optional[Union[str, Path]], optional): 作業フォルダ。
        timeout (Optional[int], optional): 1回の実行のタイムアウト秒。
        settings (Optional[Dict[str, Any]], optional): "agent" 設定ブロック。
        on_step (Optional[Callable[[Dict[str, Any]], None]], optional):
            各ステップの実行後に呼ばれるコールバック。進捗表示に使えます。

    Returns:
        Dict[str, Any]: 次のキーを持つ辞書。
            - "status": "done" / "max_steps" / "error"
            - "answer": 最終回答（未完の場合は最後の出力）
            - "steps": コードと実行結果の履歴
            - "model": 使用したモデル名
            - "message": 補足メッセージ（無ければNone）
    """
    if settings is None:
        settings = load_agent_settings()
    if max_steps is None:
        max_steps = settings.get("max_steps", 8)
    if execute is None:
        execute = settings.get("execute", "subprocess")
    if timeout is None:
        timeout = settings.get("timeout", 60)
    if work_dir is None:
        work_dir = settings.get("work_dir", "workspace")

    directory = _resolve_work_dir(work_dir)

    wait_for_ollama()
    if model is None:
        model = resolve_agent_model(settings)
    logger.info(f"エージェントモデル: {model}")

    functions_text = available_functions()
    history: List[Dict[str, Any]] = []
    last_reply: Optional[str] = None

    for step in range(1, max_steps + 1):
        prompt = build_prompt(instruction, history, functions_text, directory)
        reply, _elapsed = generate(prompt, model=model, options={"temperature": 0.2})

        if not reply:
            return {
                "status": "error",
                "answer": None,
                "steps": history,
                "model": model,
                "message": "モデルから応答が得られませんでした。",
            }
        last_reply = reply

        code = _extract_code(reply)
        if code:
            logger.info(f"ステップ{step}: コードを実行します（方式: {execute}）。")
            result = execute_code(
                code, mode=execute, work_dir=directory, step=step, timeout=timeout
            )
            record = {"step": step, "code": code, "result": result}
            history.append(record)
            if on_step:
                on_step(record)
            continue

        final = _extract_final(reply) or reply.strip()
        logger.info(f"ステップ{step}: 最終回答として確定します。")
        return {
            "status": "done",
            "answer": final,
            "steps": history,
            "model": model,
            "message": None,
        }

    return {
        "status": "max_steps",
        "answer": last_reply,
        "steps": history,
        "model": model,
        "message": f"{max_steps}ステップで完了しませんでした。",
    }


def _print_step(record: Dict[str, Any]) -> None:
    """1ステップ分の実行結果を簡潔に表示します。"""
    result = record.get("result", {})
    state = "成功" if result.get("ok") else "失敗"
    print(f"  ▶ ステップ{record['step']} 実行: {state}")
    if result.get("output"):
        print(f"    出力: {result['output'][:500]}")
    if result.get("error"):
        print(f"    エラー: {result['error'][:500]}")


def run_repl(
    model: Optional[str] = None,
    max_steps: Optional[int] = None,
    execute: Optional[str] = None,
    work_dir: Optional[Union[str, Path]] = None,
    timeout: Optional[int] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> None:
    """指示を繰り返し入力できる対話モードでエージェントを動かします。

    引数は run_agent() と同じ意味を持ちます。終了は "exit" / "quit" / "終了"
    または Ctrl+C です。
    """
    print("=== ローカルLLMエージェント ===")
    print("指示を入力すると、コードを書きながら作業します。終了は 'exit' または Ctrl+C。")

    while True:
        try:
            instruction = input("指示> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        instruction = instruction.strip()
        if not instruction:
            continue
        if instruction.lower() in ("exit", "quit", "終了"):
            break

        result = run_agent(
            instruction,
            model=model,
            max_steps=max_steps,
            execute=execute,
            work_dir=work_dir,
            timeout=timeout,
            settings=settings,
            on_step=_print_step,
        )
        print(f"[{result['status']}] {result.get('answer')}")

    print("終了します。")


def main():
    """コマンドラインの入口。引数があれば1回だけ実行し、無ければ対話モードにします。"""
    args = sys.argv[1:]
    if args:
        instruction = " ".join(args)
        result = run_agent(instruction, on_step=_print_step)
        if result.get("answer"):
            print(result["answer"])
        else:
            print(f"[{result['status']}] {result.get('message')}")
    else:
        run_repl()


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    main()
