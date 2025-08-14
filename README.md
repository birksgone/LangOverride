# Hero Skill Data Processor

## 1. プロジェクト概要

### 1.1. 目的
複数のJSONファイルとCSVファイルをデータソースとし、ヒーローごとの複雑なスキル情報（アクティブスキル、パッシブスキル、召喚物スキルなど）を再帰的に解決・統合し、最終的に構造化されたデータを出力するPythonスクリプト。

### 1.2. 成果物
- **`hero_skill_output.csv`**: 各ヒーローのスキル説明文などを解析し、フラットな形式で出力したCSVファイル。
- **`debug_hero_data.json`**: 各ヒーローについて、関連する全てのデータソースをID連鎖で解決・統合した、完全なJSONデータ。後続の解析処理における「真実の源(Single Source of Truth)」となる。

## 2. データソース

全てのデータファイルはローカルの `D:\RED` ディレクトリに配置されていることを前提とする。

-   **`characters.json`**: 全ヒーローの基本データ。全ての処理の起点となる。
-   **`specials.json`**: アクティブスキルの詳細データ (`characterSpecials`, `specialProperties`)。
-   **`battle.json`**: 戦闘関連データ。ステータス効果 (`statusEffects`)、召喚物 (`familiars`, `familiarEffects`)、パッシブスキル (`passiveSkills`) などを含む。
-   **`*_private_heroes_*.csv`**: **最重要の外部データ**。ヒーローごとの最終的なステータス（コスチュームボーナス適用済み `Max level: Attack` など）が格納されている。DoTダメージなど、ヒーロー自身の能力値を参照する計算で必須。
-   **言語ファイル群**:
    -   `English.csv` / `Japanese.csv`: スキル説明文などのテンプレート。
    -   `languageOverrides.json`: 上記言語データの上書き。

-   **`config.json`**: **手動設定ファイル**。自動推測が困難なプレースホルダーの解決ルールを定義する「例外ルール辞書」(`EXCEPTION_RULES`)。

## 3. データ処理アーキテクチャ

### 3.1. 初期化 (`main` 関数)
1.  **ゲームデータの読み込み (`load_game_data`)**: `specials.json` や `battle.json` などを読み込み、IDをキーにした辞書 `game_db` を構築する。さらに、ID検索を高速化するため、関連テーブルをすべてマージした `master_db` を作成する。
2.  **ヒーローステータスの読み込み (`load_hero_stats_from_csv`)**: `*_private_heroes_*.csv` から、ヒーローの最終ステータスDB `hero_stats_db` を構築する。
3.  **言語データと設定の読み込み**: `load_languages` で言語DBを、`load_config` で `EXCEPTION_RULES` をそれぞれメモリにロードする。

### 3.2. データ統合 (`process_all_heroes` -> `get_full_hero_data`)
- 各ヒーローをループ処理し、`get_full_hero_data` 関数を呼び出す。
- **`get_full_hero_data` の再帰的解決ロジック**:
    1. ヒーローの基本データから開始する。
    2. データ内の `specialId` や、`properties`, `passiveSkills` などのリストに含まれるIDをすべて見つけ出す。
    3. 見つけたIDを使い、`master_db` から対応するデータブロックを引く。
    4. 引いてきたデータブロックの中に、さらにIDがあれば、**再帰的にこのプロセスを繰り返し**、IDがなくなるまで全ての関連データを掘り起こす。
    5. 最終的に、すべての関連情報がマージされた、ヒーロー一人分の完全なデータオブジェクト `full_hero_data` を構築する。
- この `full_hero_data` を全ヒーロー分集めたものが `debug_hero_data.json` として出力される。

### 3.3. スキル解析 (未実装/今後の課題)
- `process_all_heroes` は、構築した `full_hero_data` を各専門パーサー (`parse_properties`, `parse_status_effects` など) に渡す。
- 各パーサーは、`full_hero_data` の中から自分が担当すべきスキルブロックを見つけ出し、以下の処理を行う。
    1. **言語IDのマッチング (`find_best_lang_id`)**: スキルブロックの `propertyType` や `statusEffect` などのキーワードを最優先し、最も確からしい説明文テンプレートのIDを特定する。
    2. **パラメータ解決 (`find_value_for_placeholder`)**:
        - まず `config.json` の `EXCEPTION_RULES` に手動ルールがないか確認する。
        - なければ、スキルブロック内に「主要な数値パラメータ」が1種類しかないかをチェックし、そうであれば無条件で採用する。
        - それでも解決できない場合は、プレースホルダー名とキー名のキーワードマッチングで値を推測する。
- 最終的に、パラメータが埋め込まれたスキル説明文を生成し、`hero_skill_output.csv` に出力する。