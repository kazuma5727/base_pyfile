@echo off
setlocal enableextensions
cd /d "%~dp0"

REM ============================================================
REM  base_pyfile クリップボードAI ランチャー
REM
REM  使い方:
REM    ai_clipboard.bat -honyaku      … コピーした内容を日本語へ翻訳
REM    ai_clipboard.bat -gojidatuji   … コピーした内容の誤字脱字を校正
REM    ai_clipboard.bat -kaitou       … コピーした内容をAIへ振り分けて回答
REM    ai_clipboard.bat -honyaku en   … 翻訳先の言語を指定（既定は ja=日本語）
REM
REM  起動後は常駐し、クリップボードが更新されるたびに自動で処理して
REM  結果をクリップボードへ書き戻します。そのまま Ctrl+V で貼り付け可。
REM  停止は Ctrl+C。
REM ============================================================

REM 先頭の「-」を外して比較する（-honyaku でも honyaku でも可）
set "ARG=%~1"
if "%ARG:~0,1%"=="-" set "ARG=%ARG:~1%"

set "MODE="
if /i "%ARG%"=="honyaku"    set "MODE=translate"
if /i "%ARG%"=="gojidatuji" set "MODE=correct"
if /i "%ARG%"=="kaitou"     set "MODE=answer"
if /i "%ARG%"=="answer"     set "MODE=answer"

if /i "%ARG%"=="help" goto :usage
if /i "%ARG%"=="h"    goto :usage

if not defined MODE (
    echo.
    echo オプションを指定してください。
    goto :usage
)

REM 翻訳先の言語（2つ目の引数。既定は日本語）
set "LANG=%~2"

REM Python の実行ファイルを探す（PATH上の python → py ランチャーの順）
set "PY=python"
where python >nul 2>nul || set "PY=py"

echo ============================================================
if "%MODE%"=="translate" echo  クリップボード 翻訳ツール（日本語へ）
if "%MODE%"=="correct"   echo  クリップボード 誤字脱字 校正ツール
if "%MODE%"=="answer"    echo  クリップボードAI 振り分けツール
echo ============================================================
echo  何かをコピーすると自動で処理し、結果をクリップボードへ書き戻します。
echo  そのまま Ctrl+V で貼り付けられます。停止は Ctrl+C。
echo ============================================================
echo.

"%PY%" -m base_pyfile.ai_clipboard %MODE% %LANG%

echo.
echo 終了しました。
pause
exit /b 0

:usage
echo.
echo 使い方: %~nx0 [オプション] [翻訳先の言語]
echo.
echo   -honyaku      コピーした内容を日本語へ翻訳（既定）
echo   -gojidatuji   コピーした内容の誤字脱字を校正
echo   -kaitou       コピーした内容をAIへ振り分けて回答
echo   -help         この使い方を表示
echo.
echo  例: %~nx0 -honyaku
echo      %~nx0 -honyaku en   （英語へ翻訳）
echo.
pause
exit /b 1
