# バージョン方針

`base_pyfile` のバージョンは **Semantic Versioning 2.0.0**（`MAJOR.MINOR.PATCH`）に従います。
バージョンは機能数や作業量のカウンターではなく、**利用者への互換性の通知**です。

## 基本ルール

| 変更の種類 | 例 | 上げ方 |
| --- | --- | --- |
| 破壊的変更 | 公開名の削除・改名、引数の削除、戻り値の形や意味の変更、既定動作の変更、新しい例外、Python / OS サポートの終了 | **MAJOR**（例: 5.0.0） |
| 後方互換のある追加 | 新関数、新モジュール、オプション引数の追加、新しい extras、新しい機能 | **MINOR**（例: 4.11.0） |
| 修正のみ | バグ修正、文字化け修正、内部リファクタ、性能改善、ドキュメント/テスト（リリースに含める場合） | **PATCH**（例: 4.10.1） |

判断は次の1問で行います。

```text
既存の公開APIを使っているコードは、変更後もそのまま動く？
  ├─ No  → MAJOR
  └─ Yes → 新しい公開機能・引数・動作が増えた？
            ├─ Yes → MINOR
            └─ No  → PATCH
```

迷ったら「既存コードが動くか」を優先してください。動かない可能性があるなら MAJOR 側に倒します。

## バージョンを上げるタイミング

バージョンはコミットごとではなく、配布物（wheel / sdist）を作る**リリース単位**で上げます。

- ドキュメント・テスト・CI のみの変更は、それだけでは `_version.py` を変更しない。`CHANGELOG.md` の `[Unreleased]` に追記し、次回リリースに含める。
- ただし、それらの変更を含む wheel / sdist を配布する場合は、PATCH 以上にする。
- 公開APIに触れる変更は、コミット時点で区分を判定して `[Unreleased]` に追記し、リリース時にバージョンを確定する。

## 公開APIの範囲

SemVer の対象になる「公開API」は次です。

- 通常の wheel に含まれる `base_pyfile` の `__all__` にある名前
- README や docstring に書かれている関数・引数・戻り値・例外・既定値
- README や AGENTS.md で案内している import 経路
- `CommandResult` などの公開データクラスのフィールド

次は対象外です。

- `base_pyfile.dev`（開発中AI。実験的で、安定版 wheel にも含まれない）
- `_` で始まる関数・変数
- 内部モジュールの実装詳細
- テスト・CI・開発用スクリプト

`base_pyfile.__version__` は `base_pyfile/_version.py` の固定値です。
`__all__` の件数から自動計算しません。理由は、リポジトリ/editable と通常の wheel で
`__all__` の件数が変わり、同じコードなのにバージョンが食い違うためです。

## このリポジトリでの例

- `doctor()` / `run_command()` / `atomic_write()` の追加 → **MINOR**（4.10.0）
- 文字化けやファイルロックの不具合修正 → **PATCH**（4.10.1）
- `found=True` の戻り値を座標から bool に変える → 厳密には **MAJOR**
- `make_directory()` の既定動作変更や `write_file()` の例外化 → 厳密には **MAJOR**

過去には、リリース直後で実利用者が少ないことを理由に、MAJOR 相当の変更を
PATCH / MINOR として出した例があります（4.9.0 / 4.9.1）。
過去の番号は遡って変更しませんが、今後はこの表に従います。

## 開発中AI（`base_pyfile.dev`）

`base_pyfile.dev` は実験的です。

- 安定版 wheel には含めない
- 破壊的変更をしても安定版の MAJOR は上げない
- ただし、dev の変更が安定版の公開APIに影響する場合は、安定版のルールに従う

## リリース手順

1. `git diff` で公開APIの変更を列挙する
2. 「基本ルール」で MAJOR / MINOR / PATCH を決める
3. `base_pyfile/_version.py` を更新する
4. `CHANGELOG.md` の `[Unreleased]` を新しいバージョンと日付に移す（先頭に空の `[Unreleased]` を残す）
5. `python -m pytest -q` を実行する
6. GitHub Actions（Windows / Python 3.10・3.12）を確認する
7. コミット・push する。必要なら `vX.Y.Z` タグを付ける

`tests/test_versioning.py` は、現在の `__version__` が `CHANGELOG.md` に
記載されていることを確認します。バージョンだけ上げて changelog を忘れると CI で落ちます。

## AI・エージェント向け指示

### このリポジトリで作業するAIへ

作業前に `AGENTS.md` とこのファイルを読んでください。

- 公開APIを追加・変更・削除した場合は、必ずバージョン区分を判定する
- 判定結果は `CHANGELOG.md` の `[Unreleased]` に追記し、`_version.py` はリリース時に更新する
- ドキュメント・テスト・CI だけの変更では `_version.py` を変えず、`[Unreleased]` に追記する
- `base_pyfile.dev` は SemVer 対象外。dev だけの変更では安定版を上げない
- 迷ったら「既存コードが動くか」で判断し、動かないなら MAJOR を選ぶ
- バージョンだけ上げて changelog を忘れない（`tests/test_versioning.py` が検出する）

### 外部AIに渡すプロンプト例

別のAIやチャットに作業を依頼するときは、次の指示をそのまま渡せます。

```text
base_pyfile の変更をお願いします。
作業前に VERSIONING.md と AGENTS.md を読んでください。

1. 公開API（通常 wheel の __all__、README/docstring の関数・引数・戻り値・例外・既定値）の
   追加・変更・削除を列挙する
2. 既存の公開API利用コードが変更なしで動くかで MAJOR / MINOR / PATCH を判定する
   - 動かない: MAJOR
   - 動くが新機能: MINOR
   - 修正のみ: PATCH
3. 公開APIに触れる変更は CHANGELOG.md の [Unreleased] に追記する
   （base_pyfile/_version.py はリリース時に更新する）
4. base_pyfile.dev は実験的なので SemVer 対象外として扱う
5. python -m pytest -q を実行し、失敗したら修正する

判断に迷った場合は、変更を分離できないか検討し、破壊的変更なら MAJOR を選んでください。
```
