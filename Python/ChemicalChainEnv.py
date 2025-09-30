import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.type_aliases import TrainFreq
from stable_baselines3.common.utils import TrainFrequencyUnit
import torch
#元素の定義
ATOM_TYPE_LIST = [
    'None','H', 'Cl', 'Br', 'F', 'I', 'O', 'S', 'N', 'P', 'C',
    'K', 'Na', 'Ba', 'Ca', 'Mg', 'Cu', 'Zn', 'Ag', 'Fe', 'Al'
] 
# お邪魔元素の定義
DISTURBER_ATOM_TYPE = 'P'
# 変換用辞書
ATOM_NAME_TO_ID = {name: i for i, name in enumerate(ATOM_TYPE_LIST)}
ATOM_ID_TO_NAME = {i: name for i, name in enumerate(ATOM_TYPE_LIST)}
# 金属グループの定義
ALKALI_METALS = {"Na", "K"}
ALKALINE_EARTH_METALS = {"Mg", "Ca", "Ba"}
TRANSITION_METALS = {"Fe", "Cu", "Zn", "Ag"}
SIMPLE_METALS = {"Al"}
# 連鎖得点の定義
CHAIN_POINTS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]
# コンボ得点の定義
COMBO_POINTS = [1, 8, 16, 32, 64, 128]
# お邪魔元素の最大数
MAX_DISTURBER = 60 
# 全消しポイント
ALL_CLEAR_POINT = 10
# ゲームオーバーペナルティ
GAME_OVER_PENALTY = -1000
# 常の行動不可のペナルティ
EVERY_ACTION_FAIL_PENALTY = -1000
# 行動不可のペナルティ
ACTION_FAIL_PENALTY = -4
# 行動ポイント
ACTION_POINT = 4

def get_atom_group(atom_name): # 元素のグループを取得
    if atom_name in ALKALI_METALS:
        return "alkali"
    elif atom_name in ALKALINE_EARTH_METALS:
        return "alkaline_earth"
    elif atom_name in TRANSITION_METALS:
        return "transition"
    elif atom_name in SIMPLE_METALS:
        return "simple"
    return "other"

class FormulaObject: # 化学式オブジェクト
    def __init__(self, name, formula, atom_dict):
        self.name = name                  # 日本語名
        self.formula = formula            # 表示用の化学式
        self.atom_dict = atom_dict        # {int: 個数}
        # 合計個数、一致判定に使用
        self.atom_count = sum(atom_dict.values())
         # 得点計算（グループに応じた倍率）
        self.point = 0
        for atom_id, count in atom_dict.items():
            atom_name = ATOM_ID_TO_NAME[atom_id]
            group = get_atom_group(atom_name)
            multiplier = 3 if group in {"alkali", "alkaline_earth", "transition", "simple"} else 1
            self.point += count * multiplier

class ChemicalEnv(gym.Env): # 環境クラス
    def __init__(self):
        super().__init__() #初期化
        self.width = 6 # フィールドの横幅
        self.height = 14 # 高さ
        self.num_atom_types = len(ATOM_TYPE_LIST)  # H, O, C, Cl, P など
        self.drop_points = [0] * self.width # 各xの高さ
        self.action_space = spaces.Discrete(self.width * 4 - 2)  # 横位置×回転4種 0: 縦向き軸下, 1: 横向き軸右, 2: 縦向き軸上, 3: 横向き軸左, 端の回転不可分2
        self.formulas = [
            FormulaObject("フッ化水素", "HF", {ATOM_NAME_TO_ID["H"]: 1, ATOM_NAME_TO_ID["F"]: 1}),
            FormulaObject("塩化水素", "HCl", {ATOM_NAME_TO_ID["H"]: 1, ATOM_NAME_TO_ID["Cl"]: 1}),
            FormulaObject("臭化水素", "HBr", {ATOM_NAME_TO_ID["H"]: 1, ATOM_NAME_TO_ID["Br"]: 1}),
            FormulaObject("ヨウ化水素", "HI", {ATOM_NAME_TO_ID["H"]: 1, ATOM_NAME_TO_ID["I"]: 1}),
            FormulaObject("塩化ナトリウム", "NaCl", {ATOM_NAME_TO_ID["Na"]: 1, ATOM_NAME_TO_ID["Cl"]: 1}),
            FormulaObject("塩化カリウム", "KCl", {ATOM_NAME_TO_ID["K"]: 1, ATOM_NAME_TO_ID["Cl"]: 1}),
            FormulaObject("塩化銀", "AgCl", {ATOM_NAME_TO_ID["Ag"]: 1, ATOM_NAME_TO_ID["Cl"]: 1}),
            FormulaObject("硫化銅", "CuS", {ATOM_NAME_TO_ID["Cu"]: 1, ATOM_NAME_TO_ID["S"]: 1}),
            FormulaObject("硫化鉄", "FeS", {ATOM_NAME_TO_ID["Fe"]: 1, ATOM_NAME_TO_ID["S"]: 1}),
            FormulaObject("硫化亜鉛", "ZnS", {ATOM_NAME_TO_ID["Zn"]: 1, ATOM_NAME_TO_ID["S"]: 1}),
            FormulaObject("硫化カルシウム", "CaS", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["S"]: 1}),
            FormulaObject("酸化マグネシウム", "MgO", {ATOM_NAME_TO_ID["Mg"]: 1, ATOM_NAME_TO_ID["O"]: 1}),
            FormulaObject("酸化カルシウム", "CaO", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["O"]: 1}),
            FormulaObject("酸化亜鉛", "ZnO", {ATOM_NAME_TO_ID["Zn"]: 1, ATOM_NAME_TO_ID["O"]: 1}),
            FormulaObject("一酸化窒素", "NO", {ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["O"]: 1}),
            FormulaObject("一酸化炭素", "CO", {ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 1}),
            FormulaObject("水", "H₂O", {ATOM_NAME_TO_ID["H"]: 2, ATOM_NAME_TO_ID["O"]: 1}),
            FormulaObject("硫化水素", "H₂S", {ATOM_NAME_TO_ID["H"]: 2, ATOM_NAME_TO_ID["S"]: 1}),
            FormulaObject("二酸化炭素", "CO₂", {ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 2}),
            FormulaObject("塩化マグネシウム", "MgCl₂", {ATOM_NAME_TO_ID["Mg"]: 1, ATOM_NAME_TO_ID["Cl"]: 2}),
            FormulaObject("塩化カルシウム", "CaCl₂", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["Cl"]: 2}),
            FormulaObject("塩化亜鉛", "ZnCl₂", {ATOM_NAME_TO_ID["Zn"]: 1, ATOM_NAME_TO_ID["Cl"]: 2}),
            FormulaObject("塩化銅(II)", "CuCl₂", {ATOM_NAME_TO_ID["Cu"]: 1, ATOM_NAME_TO_ID["Cl"]: 2}),
            FormulaObject("硫化ナトリウム", "Na₂S", {ATOM_NAME_TO_ID["Na"]: 2, ATOM_NAME_TO_ID["S"]: 1}),
            FormulaObject("水酸化ナトリウム", "NaOH", {ATOM_NAME_TO_ID["Na"]: 1, ATOM_NAME_TO_ID["O"]: 1, ATOM_NAME_TO_ID["H"]: 1}),
            FormulaObject("水酸化カリウム", "KOH", {ATOM_NAME_TO_ID["K"]: 1, ATOM_NAME_TO_ID["O"]: 1, ATOM_NAME_TO_ID["H"]: 1}),
            FormulaObject("塩化バリウム", "BaCl₂", {ATOM_NAME_TO_ID["Ba"]: 1, ATOM_NAME_TO_ID["Cl"]: 2}),
            FormulaObject("二酸化窒素", "NO₂", {ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["O"]: 2}),
            FormulaObject("二酸化硫黄", "SO₂", {ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 2}),
            FormulaObject("過酸化水素", "H₂O₂", {ATOM_NAME_TO_ID["H"]: 2, ATOM_NAME_TO_ID["O"]: 2}),
            FormulaObject("アンモニア", "NH₃", {ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["H"]: 3}),
            FormulaObject("塩化アルミニウム", "AlCl₃", {ATOM_NAME_TO_ID["Al"]: 1, ATOM_NAME_TO_ID["Cl"]: 3}),
            FormulaObject("硝酸", "HNO₃", {ATOM_NAME_TO_ID["H"]: 1, ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("硝酸銀", "AgNO₃", {ATOM_NAME_TO_ID["Ag"]: 1, ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("硝酸ナトリウム", "NaNO₃", {ATOM_NAME_TO_ID["Na"]: 1, ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("硝酸カリウム", "KNO₃", {ATOM_NAME_TO_ID["K"]: 1, ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("炭酸マグネシウム", "MgCO₃", {ATOM_NAME_TO_ID["Mg"]: 1, ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("炭酸カルシウム", "CaCO₃", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("炭酸バリウム", "BaCO₃", {ATOM_NAME_TO_ID["Ba"]: 1, ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("水酸化カルシウム", "Ca(OH)₂", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["O"]: 2, ATOM_NAME_TO_ID["H"]: 2}),
            FormulaObject("水酸化バリウム", "Ba(OH)₂", {ATOM_NAME_TO_ID["Ba"]: 1, ATOM_NAME_TO_ID["O"]: 2, ATOM_NAME_TO_ID["H"]: 2}),
            FormulaObject("酸化アルミニウム", "Al₂O₃", {ATOM_NAME_TO_ID["Al"]: 2, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("酸化鉄(III)", "Fe₂O₃", {ATOM_NAME_TO_ID["Fe"]: 2, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("メタン", "CH₄", {ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["H"]: 4}),
            FormulaObject("炭酸", "H₂CO₃", {ATOM_NAME_TO_ID["H"]: 2, ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("塩化アンモニウム", "NH₄Cl", {ATOM_NAME_TO_ID["N"]: 1, ATOM_NAME_TO_ID["H"]: 4, ATOM_NAME_TO_ID["Cl"]: 1}),
            FormulaObject("硫酸銅(II)", "CuSO₄", {ATOM_NAME_TO_ID["Cu"]: 1, ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("硫酸鉄(II)", "FeSO₄", {ATOM_NAME_TO_ID["Fe"]: 1, ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("硫酸カルシウム", "CaSO₄", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("炭酸ナトリウム", "Na₂CO₃", {ATOM_NAME_TO_ID["Na"]: 2, ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
            FormulaObject("エチレン", "C₂H₄", {ATOM_NAME_TO_ID["C"]: 2, ATOM_NAME_TO_ID["H"]: 4}),
            FormulaObject("硫酸", "H₂SO₄", {ATOM_NAME_TO_ID["H"]: 2, ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("硫酸ナトリウム", "Na₂SO₄", {ATOM_NAME_TO_ID["Na"]: 2, ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("硫酸カリウム", "K₂SO₄", {ATOM_NAME_TO_ID["K"]: 2, ATOM_NAME_TO_ID["S"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("水酸化アルミニウム", "Al(OH)₃", {ATOM_NAME_TO_ID["Al"]: 1, ATOM_NAME_TO_ID["O"]: 3, ATOM_NAME_TO_ID["H"]: 3}),
            FormulaObject("酢酸", "CH₃COOH", {ATOM_NAME_TO_ID["C"]: 2, ATOM_NAME_TO_ID["H"]: 4, ATOM_NAME_TO_ID["O"]: 2}),
            FormulaObject("リン酸", "H₃PO₄", {ATOM_NAME_TO_ID["H"]: 3, ATOM_NAME_TO_ID["P"]: 1, ATOM_NAME_TO_ID["O"]: 4}),
            FormulaObject("酢酸ナトリウム", "CH₃COONa", {ATOM_NAME_TO_ID["C"]: 2, ATOM_NAME_TO_ID["H"]: 3, ATOM_NAME_TO_ID["O"]: 2, ATOM_NAME_TO_ID["Na"]: 1}),
            FormulaObject("エタン", "C₂H₆", {ATOM_NAME_TO_ID["C"]: 2, ATOM_NAME_TO_ID["H"]: 6}),
            FormulaObject("硝酸銅(II)", "Cu(NO₃)₂", {ATOM_NAME_TO_ID["Cu"]: 1, ATOM_NAME_TO_ID["N"]: 2, ATOM_NAME_TO_ID["O"]: 6}),
            FormulaObject("硝酸カルシウム", "Ca(NO₃)₂", {ATOM_NAME_TO_ID["Ca"]: 1, ATOM_NAME_TO_ID["N"]: 2, ATOM_NAME_TO_ID["O"]: 6}),
            FormulaObject("プロピレン", "C₃H₆", {ATOM_NAME_TO_ID["C"]: 3, ATOM_NAME_TO_ID["H"]: 6}),
            FormulaObject("プロパン", "C₃H₈", {ATOM_NAME_TO_ID["C"]: 3, ATOM_NAME_TO_ID["H"]: 8}),
            FormulaObject("炭酸アンモニウム", "(NH₄)₂CO₃", {ATOM_NAME_TO_ID["N"]: 2, ATOM_NAME_TO_ID["H"]: 8, ATOM_NAME_TO_ID["C"]: 1, ATOM_NAME_TO_ID["O"]: 3}),
        ]
        #すべての状態定義
        self.observation_space = spaces.Dict(
            {
                "field": spaces.Box(low=0, high=self.num_atom_types-1,
                                    shape=(self.width, self.height), dtype=np.uint8),
                "current_pair": spaces.Box(low=0, high=self.num_atom_types -1,
                                        shape=(2,), dtype=np.uint8),
                "next_pair1": spaces.Box(low=0, high=self.num_atom_types - 1,
                                        shape=(2,), dtype=np.uint8),
                "next_pair2": spaces.Box(low=0, high=self.num_atom_types - 1,
                                        shape=(2,), dtype=np.uint8),
                "disturber_num": spaces.Box(low=0, high=MAX_DISTURBER, shape=(1,), dtype=np.uint8),
            })
        #重み付き元素の定義        
        self.atom_types = np.arange(1, self.num_atom_types)  # 元素のID
        self.atom_types = np.delete(self.atom_types, ATOM_NAME_TO_ID[DISTURBER_ATOM_TYPE])
        self.atom_weights = self._init_weights()
        #お邪魔元素の定義
        self.disturber_type = ATOM_NAME_TO_ID[DISTURBER_ATOM_TYPE]
        self.disturber_x = 0 # お邪魔元素のx座標
        self.disturber_num = 0 # お邪魔元素の数
        self.step_count = 0 # ステップ数
        #初期化
        self.reset()
    
    def _init_weights(self): #重み付き元素の定義
        atom_weights = np.zeros(self.num_atom_types, dtype=np.uint8) # 重みの初期化
        # 化学式に使われている元素の数をカウント
        for formula in self.formulas:
            for atom_id, count in formula.atom_dict.items():
                atom_weights[atom_id] += count
        # 重みを計算
        for i in range(self.num_atom_types):            
            if atom_weights[i] > 60:
                atom_weights[i] = 32
            elif atom_weights[i] > 13:
                atom_weights[i] = 16
            elif atom_weights[i] > 8:
                atom_weights[i] = 8
            elif atom_weights[i] > 3:
                atom_weights[i] = 4
            elif atom_weights[i] > 2:
                atom_weights[i] = 2
            else:
                atom_weights[i] = 1
        # 各元素の重みをセット
        atom_weights = np.array([atom_weights[i] for i in self.atom_types], dtype=np.float32)
        # 重みを正規化
        atom_weights = atom_weights / np.sum(atom_weights)
        return atom_weights

    def reset(self, *, seed=None, options=None): #初期状態
        super().reset(seed=seed)
        self.field = np.zeros((self.width, self.height), dtype=np.uint8) #何もないフィールド
        self.current_pair = self._generate_pair() #現在の元素ペアを生成
        self.next_pair1 = self._generate_pair() #次の元素ペアの生成
        self.next_pair2 = self._generate_pair() #次々元素ペアの生成
        self.disturber_x = 0 # お邪魔元素のx座標
        self.disturber_num = 0 # お邪魔元素の数
        self.drop_points = [0] * self.width # 各xの高さ
        self.disturber_point = 0 # お邪魔元素のポイント
        return self._get_obs(), {}

    def step(self, action): #次の状態へ
        #終了判定
        terminated = False
        truncated = False
        info = {} #追加情報          
        self.formula_log = [] # 化学式のログ

        # 行動を展開
        x, rot = self._develop_action(action)
        
        # 回転方向のオフセットを取得
        dx2, dy2 = self._rotation_offset(rot)
        x2 = x + dx2

        # フィールド外かどうか
        if not self._is_valid_action_x(x, x2):
            # 無効な操作なので無視 and ペナルティ
            print(f"Invalid action: x={x}, x2={x2}, rot={rot}")
            return self._fail_step(EVERY_ACTION_FAIL_PENALTY, x, x2)
        
        y, y2 = self._pair_to_position(x, x2, dy2) # ペアを配置する位置を取得
        
        # 元素があり移動できない場所かどうか
        if not self._is_valid_action_high_N(x, x2):
            # 無効な操作なので無視 and ペナルティ
            return self._fail_step(ACTION_FAIL_PENALTY, x, x2)  
        
        # フィールド外かどうか
        if not self._is_valid_action_y(y, y2):
            # 無効な操作なので無視 and ペナルティ
            print(f"Invalid action: y={y}, y2={y2}, rot={rot}")
            return self._fail_step(EVERY_ACTION_FAIL_PENALTY, x, x2)

        # 元素ペアをフィールドに配置
        self._pair_to_field(x, y, x2, y2)
        
        reward = 0 # 報酬の初期化
        comb_count = 0 # 連鎖数の初期化
        # 化学式判定
        while True:
            # 元素の組み合わせを確認
            if not self._check_formula():
                break
            # 合計得点を計算
            reward += self._count_total_points(comb_count)
            comb_count += 1 # 連鎖数をカウント
            # 元素を削除する
            self._delete_atoms()
            # 自由落下を行う
            self._free_fall()

        # 全消しボーナス
        # 最下段（y=0）がすべて空ならボーナス
        if np.all(self.field[:, 0] == 0):
            reward += ALL_CLEAR_POINT
            # 一番下の行をOで埋める
            self.field[:, 0] = ATOM_NAME_TO_ID["O"]

        # 毎ターンの報酬
        reward += ACTION_POINT

        # (2,11)が埋まったらゲームオーバーにする
        if self.field[2, 11] != 0:
            terminated = True
            reward = GAME_OVER_PENALTY

        # じゃまな元素追加
        self._set_disturber()
        # 化学式判定(報酬なし)
        while True:
            # 元素の組み合わせを確認
            if not self._check_formula():
                break
            # 元素を削除する
            self._delete_atoms()
            # 自由落下を行う
            self._free_fall()

        # (2,11)が埋まったらゲームオーバーにする
        if self.field[2, 11] != 0:
            terminated = True

        #元素ペアの生成
        self.current_pair = self.next_pair1
        self.next_pair1 = self.next_pair2
        self.next_pair2 = self._generate_pair()

        # お邪魔元素の数を更新
        self._add_disturber()

        obs = self._get_obs() #観測値
        info = self._build_info(reward, False, x, x2) #追加情報
        return obs, reward, terminated, truncated, info
    
    # def _is_valid_action_high(self, x, x2): #回転によってたどり着けるか
    #     if(x < 2 or x2 < 2):
    #         if(self.drop_points[1] >= 12): #下にある元素が12以上のとき左には行けない
    #             return False
    #     if(x < 1 or x2 < 1):
    #         if(self.drop_points[0] >= 12 and self.drop_points[0] - self.drop_points[1] > 1): #階段状ならOK
    #             return False
    #     if(x > 2 or x2 > 2):    
    #         if(self.drop_points[3] >= 12): #下にある元素が12以上のとき右には行けない
    #             return False
    #     if(x > 3 or x2 > 3):
    #         if(self.drop_points[4] >= 12 and self.drop_points[4] - self.drop_points[3] > 1): #階段状ならOK
    #             return False
    #     if(x > 4 or x2 > 4):
    #         if(self.drop_points[5] >= 12 and self.drop_points[5] - self.drop_points[4] > 1 and self.drop_points[4] - self.drop_points[3] > 1): #階段状ならOK
    #             return False
    #     return True    
    def _is_valid_action_high_N(self, x, x2):
        if(x < 2 or x2 < 2):
            if(self.drop_points[1] > 12): #下にある元素が12以上のとき左には行けない
                return False
        if(x < 1 or x2 < 1):
            if(self.drop_points[0] > 12): #下にある元素が12以上のとき左には行けない
                return False
        if(x > 2 or x2 > 2):    
            if(self.drop_points[3] > 12): #下にある元素が12以上のとき右には行けない
                return False
        if(x > 3 or x2 > 3):
            if(self.drop_points[4] > 12): #下にある元素が12以上のとき右には行けない
                return False
        if(x > 4 or x2 > 4):
            if(self.drop_points[5] > 12): #下にある元素が12以上のとき右には行けない
                return False
        return True    
    def _is_valid_action_x(self, x, x2): # フィールドの範囲内か
        # フィールドの範囲内か
        if x2 < 0 or x2 >= self.width:
            return False
        return True     
    
    def _is_valid_action_y(self, y, y2): # フィールドの範囲内か
        # フィールドの範囲内か
        if y < 0 or y >= self.height:
            return False
        if y2 < 0 or y2 >= self.height:
            return False 
        return True     

    def render(self, mode="human"):
        print(self.field[::-1])  # 上が上になるように表示

    def _generate_pair(self): # 元素ペアを生成
        # 重み付き元素を使用してペアを生成
        atom_types = np.random.choice(self.atom_types, size=2, replace=True, p=self.atom_weights)
        return atom_types
    
    def _get_obs(self): # 観測値を取得
        obs = {
            "field": self.field.copy(),  # フィールドの盤面
            "current_pair": np.array(self.current_pair, dtype=np.uint8), #現在の元素ペア
            "next_pair1": np.array(self.next_pair1, dtype=np.uint8), #次の元素ペア
            "next_pair2": np.array(self.next_pair2, dtype=np.uint8), #次々元素ペア
            "disturber_num": np.array([self.disturber_num], dtype=np.uint8)
        }
        return obs
    
    def _develop_action(self, action): # 行動を展開
        # 行動不可を考慮して、回転方向とx座標を取得
        action += 1 # 端の回転不可分2を考慮
        x = action % self.width
        rot = action // self.width
        return x, rot
    
    def _rotation_offset(self, rot): # 回転方向のオフセットを取得
        # 行動不可を考慮して、回転方向に応じたオフセットを返す
        if rot == 1: # 縦向き軸下
            return 0, 1
        elif rot == 0:  # 横向き軸右
            return -1, 0
        elif rot == 2:  # 縦向き軸上
            return 0, -1
        elif rot == 3:  # 横向き軸左
            return 1, 0
        return 0, 1
    
    def _pair_to_field(self, x, y, x2, y2): # フィールドにペアを配置するロジック
            self.field[x, y] = self.current_pair[0]
            self.field[x2, y2] = self.current_pair[1]
            # drop_pointを更新
            self.drop_points[x] = y + 1
            self.drop_points[x2] = y2 + 1

    def _pair_to_position(self, x, x2, dy2): # ペアを配置する位置
        y = self.drop_points[x]
        y2 = self.drop_points[x2] + dy2
        if 0 > dy2: # 軸が上にあるとき
            y = self.drop_points[x] + 1
            y2 = self.drop_points[x2]        
        return y, y2

    def _check_formula(self): # 元素の組み合わせを確認するロジック
        success = False
        self.atom_object_hashs = {}
        #すべての化学式を探索
        for formula in self.formulas:
            #化学式が成立しているか
            if self._check_connection(formula):
                success = True
        return success
    
    def _count_total_points(self, comb_count): # 合計得点を計算するロジック
        total_point = 0 # 合計得点
        chain_count = 0 # 連鎖数

        # 得点の低いものから処理
        sorted_formulas = sorted(
            self.atom_object_hashs.items(),
            key=lambda kv: kv[0].point * len(kv[1])
        )

        for formula, matched_positions in sorted_formulas:
            # 一致した原子の数をカウント
            matched_count = len(matched_positions)
            # 基礎得点
            base_point = formula.point
            # 連鎖数に応じた倍率
            chain_rate = CHAIN_POINTS[len(CHAIN_POINTS) - 1] if chain_count >= len(CHAIN_POINTS) else CHAIN_POINTS[chain_count]
            # コンボ数に応じた倍率
            combo_rate = COMBO_POINTS[len(COMBO_POINTS) - 1] if comb_count >= len(COMBO_POINTS) else COMBO_POINTS[comb_count]
            # 得点計算
            score = matched_count * base_point * chain_rate * combo_rate
            total_point += score

            # 揃えた化学式と原子の個数のログを記録
            self.formula_log.append((formula.name, matched_count))

            # デバッグ用出力
            #print(f"[{formula.name}] {matched_count} match × {base_point} pt × {chain_rate} chain × {combo_rate} combo = {score} pt")
            # 連鎖数をカウント
            chain_count += 1

        return total_point
    
    def _delete_atoms(self): # 原子を削除するロジック
        for formula, positions in self.atom_object_hashs.items():
            for x, y in positions:
                if self.field[x][y] != 0:
                    # 原子を削除
                    self.field[x][y] = 0
                    if self.drop_points[x] > y:
                        self.drop_points[x] = y
    
    def _free_fall(self): # 自由落下を行うロジック
        for x in range(self.width):
            # 自由落下を行う
            for y in range(self.drop_points[x], self.height):
                if self.field[x][y] != 0:
                    if self.drop_points[x] < y:
                        self.field[x][self.drop_points[x]] = self.field[x][y]
                        self.field[x][y] = 0
                    self.drop_points[x] +=  1

    def _check_connection(self, formula): #化学式が成立しているかを探索し、バッファに追加
        existed = False
        width = self.width
        height = self.height
        visited = [False] * (width * height) # 比較したマスを記録するリスト
        # フィールドの全てのマスを探索
        for x in range(width):
            for y in range(height):
                # フィールドに元素があるか、またはすでに揃っている原子に含まれているか
                if self.field[x][y] != 0 and (
                    not existed or (existed and (x, y) not in self.atom_object_hashs.get(formula, set()))
                ):
                    required_atoms = dict(formula.atom_dict)  # AtomType -> count
                    check_buffer = set() # 該当した原子のマスを記録するバッファ
                    visited[:] = [False] * (width * height) # 初期化
                    # 探索を開始
                    if self._explore_connections(x, y, visited, required_atoms, formula.atom_count, width, height, check_buffer):
                        # 初めて成立したらセットを初期化
                        if formula not in self.atom_object_hashs:
                            self.atom_object_hashs[formula] = set()
                        self.atom_object_hashs[formula].update(check_buffer)
                        existed = True
        return existed
    
    def _explore_connections(self, start_x, start_y, visited, required_atoms, atom_count, width, height, check_buffer): # 探索を行う
        # スタックを使用して探索
        stack = [(start_x, start_y)]
        # スタックが空になるまで探索
        while stack:
            x, y = stack.pop()
            # フィールドの範囲外か
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # すでに訪れたマスか、または元素がない場合
            index = y * width + x
            if visited[index] or self.field[x][y] == 0:
                continue

            visited[index] = True # すでに訪れたマスを記録
            atom_type = self.field[x][y] # 元素の種類

            # 元素の種類が必要な元素の中にあるか
            if atom_type not in required_atoms or required_atoms[atom_type] <= 0:
                continue

            # 元素の種類が必要な元素の中にある場合    
            required_atoms[atom_type] -= 1
            atom_count -= 1
            check_buffer.add((x, y))

            # すべての元素が揃った場合
            if atom_count <= 0:
                return True

            # 隣接マス探索
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                stack.append((x + dx, y + dy))

        # すべての元素が揃わなかった場合
        return False
    
    def _fail_step(self, penalty, x, x2): # 処理失敗時の処理
        reward = penalty # ペナルティ
        obs = self._get_obs() # 観測値を取得
        info = self._build_info(reward, True, x, x2) # 追加情報を取得
        return obs, reward, False, False, info
   
    def _build_info(self, reward, invalid, x, x2):  # 情報の構築
        return {
            "formula_log": self.formula_log,
            "score_gained": reward,
            "disturber_num": self.disturber_num,
            "invalid_action": invalid,
            "field": self.field.copy(),
            "x": x,
            "x2": x2,
        }
    
    def _set_disturber(self): # お邪魔元素をフィールドに配置するロジック
        if self.disturber_num <= 0:
            return
        # お邪魔元素をフィールドに配置
        while True:
            for i in range(self.width):
                xi = (self.disturber_x + i) % self.width
                if self.drop_points[xi] < self.height:
                    self.field[xi][self.drop_points[xi]] = self.disturber_type
                    self.drop_points[xi] += 1
                    self.disturber_num -= 1
                    self.disturber_x = (xi + 1) % self.width
                    if(self.disturber_num <= 0):
                        return
                    break
                elif i >= self.width - 1:
                    return
    
    def _add_disturber(self): # お邪魔元素の数を追加するロジック
        #確率でお邪魔元素を追加        
        self.step_count += 1
        if self.step_count > 30 and np.random.rand() < 0.05:
            #じゃま追加
            if self.disturber_num < MAX_DISTURBER:
                self.disturber_num += np.random.randint(self.step_count // 20, self.step_count // 15 + 1)
                self.step_count = 0

    # def flatten_observation(self, obs_dict):
    #     parts = [
    #         obs_dict["field"].flatten(),
    #         obs_dict["current_pair"].flatten(),
    #         obs_dict["next_pair1"].flatten(),
    #         obs_dict["next_pair2"].flatten(),
    #         obs_dict["disturber_num"].flatten(),
    #     ]
    #     return np.concatenate(parts).astype(np.float32)
    
    def flatten_observation(obs_dict: dict[str, np.ndarray]) -> torch.Tensor:
        parts = [
            torch.from_numpy(obs_dict["field"]).reshape(-1),
            torch.from_numpy(obs_dict["current_pair"]).reshape(-1),
            torch.from_numpy(obs_dict["next_pair1"]).reshape(-1),
            torch.from_numpy(obs_dict["next_pair2"]).reshape(-1),
            torch.from_numpy(obs_dict["disturber_num"]).reshape(-1),
        ]
        return torch.cat(parts).to(dtype=torch.float32)

if __name__ == "__main__":
    os.chdir(r"C:\Users\owner\Desktop")
    
    # 環境の登録
    register(
        id="ChemicalEnv-v2",
        entry_point="ChemicalChainEnv:ChemicalEnv",
        max_episode_steps=500,
    )
    # 環境作成（並列環境も可）
    env = make_vec_env("ChemicalEnv-v2", n_envs=1)

    model = None # モデルの初期化
    model_path = "my_model4.zip"
    print(f"モデルファイルのパス: {os.path.abspath(model_path)}")
    if os.path.exists(model_path):
        model = DQN.load(model_path)
        model.set_env(env) # 既存のモデルがある場合、環境を設定
        if model is not None:
            print(f"モデル '{model_path}' を読み込みました。")
    else:
        print(f"モデルファイル '{model_path}' が見つかりませんでした。")   

    if(model == None):
        # DQNエージェントの作成
        model = DQN(
            "MultiInputPolicy",     # 使用する方策ネットワーク（MLP = 全結合層）
            env,                    # 対象の環境（VecEnv 推奨）
            verbose=1,              # ログ出力レベル（1 = progress bar 表示あり）
            learning_rate=3e-4,     # 学習率（小さすぎると遅い、大きすぎると不安定）
            buffer_size=50000,      # リプレイバッファのサイズ
            learning_starts=1000,   # 学習開始までのステップ数（最初はバッファを埋める）
            batch_size=64,          # バッチサイズ（1回の学習で使用するサンプル数）
            tau=1.0,                # ソフトターゲット更新の係数（DQNでは主に 1.0 固定）
            gamma=0.99,             # 割引率（未来の報酬の重要度）
            train_freq=4,           # 学習を何ステップごとに行うか
            target_update_interval=2000, # ターゲットネットワークの更新間隔（step単位）
            tensorboard_log="C:/Users/owner/Documents/ChemicalChain/dqn_log2", 
            exploration_fraction=1, # 探索率が減少するステップ比
            exploration_final_eps=0.1, # 最小探索率
        )
        
        # # コールバックの設定
        # eval_callback = EvalCallback(
        #     env,
        #     best_model_save_path="C:/Users/owner/Documents/ChemicalChain/best_model",
        #     log_path="C:/Users/owner/Documents/ChemicalChain/eval_logs",
        #     eval_freq=5000,        # 5000ステップごとに評価
        #     deterministic=True,
        #     render=False
        # )
        # model.exploration_rate = 0.3
        # model.target_update_interval = 1000
        # model.exploration_fraction=0.3 # 探索率が減少するステップ比
        # model.exploration_final_eps=0.05 # 最小探索率
        # # model.train_freq = TrainFreq(1, TrainFrequencyUnit.EPISODE)
        # print("exploration_fraction:", model.exploration_fraction, type(model.exploration_fraction))
        # print("exploration_final_eps:", model.exploration_final_eps, type(model.exploration_final_eps))
        # model.train_freq = TrainFreq(2, TrainFrequencyUnit.STEP) # 学習を1ステップごとに行う
        # 学習開始
        #model.learn(total_timesteps=1_000_000, callback=eval_callback, reset_num_timesteps=False)
        model.learn(total_timesteps=500_000, reset_num_timesteps=False)
        # 学習後のモデル保存
        model.save(model_path)

        # 学習モデルをテスト
        action_count = np.zeros(model.action_space.n, dtype=int)
