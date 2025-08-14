あなたは、Pythonによるデータ処理スクリプト開発のシニアエキスパートです。
私たちは今、ヒーローのスキルデータを解析するGUIツールの開発プロジェクトを進めています。
これまでの開発で、以下の仕様とアーキテクチャが固まり、それに基づいたPythonスクリプトの基盤が完成しています。

---
### **現状の仕様サマリー (`README.md`より抜粋)**

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

---
### **現在のコード**

All_Hero_Extract.py

---
### **直近の課題と次のステップ**

データ統合の土台 (`get_full_hero_data` による `full_hero_data` の生成) は完成し、`debug_hero_data.json` にて、`passiveSkills` を含む全関連データが正しく展開されることを確認済みです。

しかし、`process_all_heroes` から先の**解析フェーズ**が、この新しいデータ構造 (`full_hero_data`) にまだ完全に対応できていません。
その結果、`hero_skill_output.csv` が、多くのスキル情報を解析できずに、不完全な状態で出力されています。

**次に解決すべき課題は以下の通りです:**

1.  **パーサー群の再構築**:
    *   `process_all_heroes` が `full_hero_data` の中から解析すべきスキルブロック（`properties`, `statusEffects`, `familiars` など）を全て探し出し、適切なパーサーに渡すように修正する。
    *   各パーサー (`parse_properties` など) を、この新しいデータフローに対応できるように修正する。

2.  **パラメータ解決の精度向上**:
    *   `find_value_for_placeholder` の自動推測ロジックを改善し、未解決プレースホルダーの数を減らす。
    *   解決が困難なものは、`config.json` の `EXCEPTION_RULES` にルールを追加していく開発スタイルを確立する。

3.  **特殊な説明文の整形**:
    *   `fire_god_zidane` の `{STATUSEFFECTS}` のように、複数のスキル情報を箇条書きで埋め込むパターンの整形ロジックを実装する。
    *   `+` や `-` 記号を、文脈に応じて正しく付与するフォーマットルールを実装する。

**まずは、課題1「パーサー群の再構築」から着手し、`moth_pepperbleu` のFamiliarスキルが正しくCSVに出力されることを目指します。**

---
