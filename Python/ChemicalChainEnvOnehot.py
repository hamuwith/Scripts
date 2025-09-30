import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ChemicalChainEnv import ChemicalEnv, ATOM_NAME_TO_ID, FormulaObject, MAX_DISTURBER, DISTURBER_ATOM_TYPE

class ChemicalEnvOnehot(ChemicalEnv): # 環境クラス
    def __init__(self):        
        self.eye_table = None
        super().__init__() #初期化
        #すべての状態定義
        self.observation_space = spaces.Dict(
            {
                "field": spaces.Box(low=0, high=1,
                        shape=(self.num_atom_types - 1, self.width, self.height), dtype=np.float32),
                "current_pair": spaces.Box(low=0, high=1,
                           shape=(2, self.num_atom_types - 1), dtype=np.float32),                           
                "next_pair1": spaces.Box(low=0, high=1,
                           shape=(2, self.num_atom_types - 1), dtype=np.float32),                           
                "next_pair2": spaces.Box(low=0, high=1,
                           shape=(2, self.num_atom_types - 1), dtype=np.float32),
                "disturber_num": spaces.Box(low=0, high=MAX_DISTURBER, shape=(1,), dtype=np.float32),
            }) 
        
    def _get_obs(self): # 観測値を取得
        if self.eye_table is None:
            self.eye_table = np.eye(self.num_atom_types, dtype=np.float32)
        # 観測値のために各説明関数をone-hotエンコーディングで表現
        # フィールドの盤面をone-hotエンコーディング
        field = self.eye_table[self.field].transpose(2, 0, 1)[1:]
        current_pair = self.eye_table[self.current_pair][..., 1:]
        next_pair1 = self.eye_table[self.next_pair1][..., 1:]
        next_pair2 = self.eye_table[self.next_pair2][..., 1:]
        obs = {
            "field": field,  # フィールドの盤面
            "current_pair": current_pair, #現在の元素ペア
            "next_pair1": next_pair1, #次の元素ペア
            "next_pair2": next_pair2, #次々元素ペア
            "disturber_num": np.array([self.disturber_num], dtype=np.float32)
        }
        return obs