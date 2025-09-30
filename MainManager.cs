using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;
using System.Threading;
using Cysharp.Threading.Tasks;
using DG.Tweening;
using TMPro;
//未実装は'***'で表記

public class MainManager : MonoBehaviour
{
    #region シリアライズフィールド
    [SerializeField] Vector2Int size; //ゲームの幅と高さ
    [SerializeField] int bonusRate; //ボーナスの掛け率
    [SerializeField] Vector2Int startPosition; //原子開始位置
    [SerializeField] float dropTime; //落下時間
    [SerializeField] float continuousMoveTime; //連続移動時間
    [SerializeField] float downAcceleration; //下加速量
    [SerializeField] AtomObject atomPrefab; //原子プレハブ
    [SerializeField] SpriteRenderer dropPointPrefab; //落下位置プレハブ
    [SerializeField] int[] chainPointRates; //連鎖数の得点倍率
    [SerializeField] int[] comboPointRates; //コンボ数の得点倍率
    [SerializeField] AtomType disturbanceAtom; //おじゃま原子
    [SerializeField] AtomType fullClearAtom; //全消し原子
    [SerializeField] float disturbancAtomsSize; //じゃま原子のサイズ
    [SerializeField] int bonusAtomMin; //ボーナスの化学式の原子数の下限
    [SerializeField] float updateDropTime; //落下時間更新
    [SerializeField] float updateDropTimeRate; //落下時間更新率
    [SerializeField] GameObject stockAtoms; //原子ストック
    [SerializeField] float formulaRate; //揃えた化学式の得点倍率
    [SerializeField] TextMeshProUGUI countdown; //カウントダウン
    [SerializeField] RectTransform countdownRectTransform; //カウントダウン
    #endregion シリアライズフィールド

    #region 公開フィールド
    public static MainManager Instance { get; private set; } //インスタンスを保持するプロパティ
    public FormulaObject[] Formulas { get; private set; } = new FormulaObject[]
    {
        new FormulaObject("フッ化水素", "HF", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.F, 1 } }),
        new FormulaObject("塩化水素", "HCl", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("臭化水素", "HBr", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.Br, 1 } }),
        new FormulaObject("ヨウ化水素", "HI", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.I, 1 } }),
        new FormulaObject("塩化ナトリウム", "NaCl", new Dictionary<AtomType, int> { { AtomType.Na, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("塩化カリウム", "KCl", new Dictionary<AtomType, int> { { AtomType.K, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("塩化銀", "AgCl", new Dictionary<AtomType, int> { { AtomType.Ag, 1 }, { AtomType.Cl, 1 } }),
        new FormulaObject("硫化銅", "CuS", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("硫化鉄", "FeS", new Dictionary<AtomType, int> { { AtomType.Fe, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("硫化亜鉛", "ZnS", new Dictionary<AtomType, int> { { AtomType.Zn, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("硫化カルシウム", "CaS", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.S, 1 } }),
        new FormulaObject("酸化マグネシウム", "MgO", new Dictionary<AtomType, int> { { AtomType.Mg, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("酸化カルシウム", "CaO", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("酸化亜鉛", "ZnO", new Dictionary<AtomType, int> { { AtomType.Zn, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("一酸化窒素", "NO", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("一酸化炭素", "CO", new Dictionary<AtomType, int> { { AtomType.C, 1 }, { AtomType.O, 1 } }),
        new FormulaObject("水", "H<sub>2</sub>O", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.O, 1 } }),
        new FormulaObject("硫化水素", "H<sub>2</sub>S", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.S, 1 } }),
        new FormulaObject("二酸化炭素", "CO<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.C, 1 }, { AtomType.O, 2 } }),
        new FormulaObject("塩化マグネシウム", "MgCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Mg, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("塩化カルシウム", "CaCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("塩化亜鉛", "ZnCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Zn, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("塩化銅(II)", "CuCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("硫化ナトリウム", "Na<sub>2</sub>S", new Dictionary<AtomType, int> { { AtomType.Na, 2 }, { AtomType.S, 1 } }),
        new FormulaObject("水酸化ナトリウム", "NaOH", new Dictionary<AtomType, int> { { AtomType.Na, 1 }, { AtomType.O, 1 }, { AtomType.H, 1 } }),
        new FormulaObject("水酸化カリウム", "KOH", new Dictionary<AtomType, int> { { AtomType.K, 1 }, { AtomType.O, 1 }, { AtomType.H, 1 } }),
        new FormulaObject("塩化バリウム", "BaCl<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ba, 1 }, { AtomType.Cl, 2 } }),
        new FormulaObject("二酸化窒素", "NO<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.O, 2 } }),
        new FormulaObject("二酸化硫黄", "SO<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.S, 1 }, { AtomType.O, 2 } }),
        new FormulaObject("過酸化水素", "H<sub>2</sub>O<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.O, 2 } }),
        new FormulaObject("アンモニア", "NH<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.H, 3 } }),
        new FormulaObject("塩化アルミニウム", "AlCl<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Al, 1 }, { AtomType.Cl, 3 } }),
        new FormulaObject("硝酸", "HNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.H, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("硝酸銀", "AgNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Ag, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("硝酸ナトリウム", "NaNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Na, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("硝酸カリウム", "KNO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.K, 1 }, { AtomType.N, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("炭酸マグネシウム", "MgCO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Mg, 1 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("炭酸カルシウム", "CaCO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("炭酸バリウム", "BaCO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Ba, 1 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("水酸化カルシウム", "Ca(OH)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.O, 2 }, { AtomType.H, 2 } }),
        new FormulaObject("水酸化バリウム", "Ba(OH)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ba, 1 }, { AtomType.O, 2 }, { AtomType.H, 2 } }),
        new FormulaObject("酸化アルミニウム", "Al<sub>2</sub>O<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Al, 2 }, { AtomType.O, 3 } }),
        new FormulaObject("酸化鉄(III)", "Fe<sub>2</sub>O<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Fe, 2 }, { AtomType.O, 3 } }),
        new FormulaObject("メタン", "CH<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.C, 1 }, { AtomType.H, 4 } }),
        new FormulaObject("炭酸", "H<sub>2</sub>CO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("塩化アンモニウム", "NH<sub>4</sub>Cl", new Dictionary<AtomType, int> { { AtomType.N, 1 }, { AtomType.H, 4 }, { AtomType.Cl, 1 } }),
        new FormulaObject("硫酸銅(II)", "CuSO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("硫酸鉄(II)", "FeSO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Fe, 1 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("硫酸カルシウム", "CaSO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("炭酸ナトリウム", "Na<sub>2</sub>CO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Na, 2 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
        new FormulaObject("エチレン", "C<sub>2</sub>H<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 4 } }),
        new FormulaObject("硫酸", "H<sub>2</sub>SO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.H, 2 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("硫酸ナトリウム", "Na<sub>2</sub>SO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.Na, 2 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("硫酸カリウム", "K<sub>2</sub>SO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.K, 2 }, { AtomType.S, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("水酸化アルミニウム", "Al(OH)<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.Al, 1 }, { AtomType.O, 3 }, { AtomType.H, 3 } }),
        new FormulaObject("酢酸", "CH<sub>3</sub>COOH", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 4 }, { AtomType.O, 2 } }),
        new FormulaObject("リン酸", "H<sub>3</sub>PO<sub>4</sub>", new Dictionary<AtomType, int> { { AtomType.H, 3 }, { AtomType.P, 1 }, { AtomType.O, 4 } }),
        new FormulaObject("酢酸ナトリウム", "CH<sub>3</sub>COONa", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 3 }, { AtomType.O, 2 }, { AtomType.Na, 1 } }),
        new FormulaObject("エタン", "C<sub>2</sub>H<sub>6</sub>", new Dictionary<AtomType, int> { { AtomType.C, 2 }, { AtomType.H, 6 } }),
        new FormulaObject("硝酸銅(II)", "Cu(NO<sub>3</sub>)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Cu, 1 }, { AtomType.N, 2 }, { AtomType.O, 6 } }),
        new FormulaObject("硝酸カルシウム", "Ca(NO<sub>3</sub>)<sub>2</sub>", new Dictionary<AtomType, int> { { AtomType.Ca, 1 }, { AtomType.N, 2 }, { AtomType.O, 6 } }),
        new FormulaObject("プロピレン", "C<sub>3</sub>H<sub>6</sub>", new Dictionary<AtomType, int> { { AtomType.C, 3 }, { AtomType.H, 6 } }),
        new FormulaObject("プロパン", "C<sub>3</sub>H<sub>8</sub>", new Dictionary<AtomType, int> { { AtomType.C, 3 }, { AtomType.H, 8 } }),
        new FormulaObject("炭酸アンモニウム", "(NH<sub>4</sub>)<sub>2</sub>CO<sub>3</sub>", new Dictionary<AtomType, int> { { AtomType.N, 2 }, { AtomType.H, 8 }, { AtomType.C, 1 }, { AtomType.O, 3 } }),
    }; //化学式の配列
    [HideInInspector] public event Action<AtomType> onCreateAtomType; //原子生成イベント
    #endregion 公開フィールド

    #region プライベートフィールド 
    int[] atomWeight; //原子の重み
    State currentState; //現在の状態を管理する変数
    int fullWeight; //重みの合計
    PlayerBase[] players; //プレイヤー
    float gameTime; //ゲーム時間
    FormulaObject bonusFormula; //ボーナス化学式
    float updateDropCount; //落下更新カウント
    CancellationTokenSource cts; //キャンセルトークン
    readonly Color disturbanceAtomColor = new Color32(0x22, 0x22, 0x22, 255);    //おじゃま色
    static readonly Dictionary<AtomGroupType, Color> groupColors = new Dictionary<AtomGroupType, Color>
    {
        { AtomGroupType.Hydrogen,           new Color32(0x00, 0xCF, 0xFF, 255) },
        { AtomGroupType.Halogen,            new Color32(0x00, 0xAA, 0x88, 255) },
        { AtomGroupType.OxygenSulfur,       new Color32(0xA6, 0xE2, 0x2E, 255) },
        { AtomGroupType.NitrogenPhosphorus, new Color32(0xB3, 0x9D, 0xDB, 255) },
        { AtomGroupType.Carbon,             new Color32(0x44, 0x44, 0x44, 255) },
        { AtomGroupType.AlkaliMetal,        new Color32(0xFF, 0xD7, 0x00, 255) },
        { AtomGroupType.AlkalineEarthMetal, new Color32(0xD1, 0xC4, 0xE9, 255) },
        { AtomGroupType.TransitionMetal,    new Color32(0x3F, 0x51, 0xB5, 255) },
        { AtomGroupType.SimpleMetal,        new Color32(0xB0, 0xBE, 0xC5, 255) },
        { AtomGroupType.None,               Color.clear }
    }; //原子の色を管理する辞書
    #endregion プライベートフィールド

    #region 定数
    public const int NormalMax = 8; //じゃま普通表示の最大値
    #endregion 定数

    #region プロパティ
    public State CurrentState //現在の状態を取得または設定するプロパティ
    {
        get 
        {
            return currentState; 
        }
        private set 
        { 
            if(currentState == value) return; //状態が変わらない場合は何もしない
            currentState = value; 
            switch(currentState)
            {
                case State.Ready:
                    // Ready状態の処理
                    Countdown().Forget();
                    break;
                case State.Play:
                    // Play状態の処理
                    break;
                case State.Pause:
                    // Pause状態の処理
                    break;
                case State.Result:
                    // Result状態の処理
                    break;
                default:
                    break;
            }
        }
    }
    public AtomType DisturbanceAtom => disturbanceAtom; //じゃま原子種別
    public AtomType FullClearAtom => fullClearAtom; //全消し原子種別
    public int Level { get; set; } //ゲームのレベルを管理するプロパティ
    public int ChainPointRatesLength //連鎖率の長さ
    {
        get
        {
            return chainPointRates.Length;
        }
    }
    public int ComboPointRatesLength //コンボ率の長さ
    {
        get
        {
            return comboPointRates.Length;
        }
    }
    public float DisturbancAtomsSize => disturbancAtomsSize; //じゃま原子のサイズ
    public List<AtomObject> DisturbanceAtomObjects { get; set; } //おじゃま原子を管理する配列
    public int StageSize => size.x * size.y; //ステージサイズ
    public Color DisturbanceAtomColor => disturbanceAtomColor; //ステージの幅
    public SpriteRenderer DropPointPrefab => dropPointPrefab; //落下位置プレハブ
    public Vector2Int Size => size; //ゲームの幅と高さ
    public Vector2Int StartPosition => startPosition; //原子開始位置
    public float DropTime => dropTime; //落下時間
    public float ContinuousMoveTime => continuousMoveTime; //連続移動時間
    public float DownAcceleration => downAcceleration; //下加速量
    public AtomObject AtomPrefab => atomPrefab; //原子プレハブ
    public GameObject StockAtoms => stockAtoms; //原子ストック
    #endregion プロパティ

    #region 列挙体
    public enum State
    {
        None,
        Ready,
        Play,
        Pause,
        Result,
    } //ゲームの状態を管理する列挙型
    public enum AtomType
    {
        H, 
        Cl,
        Br,
        F,
        I,
        O,
        S,
        N,
        P,
        C,  
        K,
        Na,
        Ba,
        Ca,
        Mg,
        Cu,
        Zn,
        Ag,
        Fe,
        Al,
        None,
    } //原子の種類を管理する列挙型
    public enum AtomGroupType //原子のグループを管理する列挙型
    {
        Hydrogen = 1,
        Halogen,
        OxygenSulfur,
        NitrogenPhosphorus,
        Carbon,
        AlkaliMetal,
        AlkalineEarthMetal,
        TransitionMetal,
        SimpleMetal,
        None
    }
    #endregion 列挙体

    #region Unityイベント
    private void Awake()
    {
        Instance = this;
    }
    private void Start()
    {
        //ゲームの初期化
        Init();
        for(int i = 0; i < players.Length; i++)
        {
            players[i].Start0(i); // プレイヤーの初期化
        }
        CurrentState = State.Ready; //ゲームの状態を初期化
    }
    private void Update()
    {
        switch (CurrentState)
        {
            case State.Ready:
                // Ready状態の処理
                //次の原子までをセット
                foreach (var player in players)
                {
                    player.ReadyAtom(); //原子をセット
                }   
                break;
            case State.Play:
                // Play状態の処理
                UpdateGameTime(); //ゲーム時間を更新
                foreach (var player in players)
                {
                    player.UpdatePlayState(); // プレイヤーの状態を更新
                }
                break;
            case State.Pause:
                // Pause状態の処理
                break;
            case State.Result:
                // Result状態の処理
                break;
            default:
                break;
        }
    }
    private void OnDestroy()
    {
        //キャンセルトークンをキャンセル
        if (cts != null)
        {
            cts.Cancel();
            cts.Dispose();
            cts = null;
        }
    } //タスクのキャンセル
    #endregion Unityイベント

    #region 公開メソッド 
    public void CreateAtoms() //ペア原子を生成するメソッド
    {
        //重みに応じて原子の種類を決定
        int atomCount = UnityEngine.Random.Range(0, fullWeight); //原子の種類をランダムに選択
        for (int j = 0; j < (int)AtomType.None; j++)
        {
            if (atomWeight[j] > atomCount)
            {
                onCreateAtomType.Invoke((AtomType)j); //原子生成イベントを発火
                break;
            }
            atomCount -= atomWeight[j];
        }
    }
    public void CalcPoint(ref PointSet pointSet, int atomCount,　int formulaPoint, int chainCount, int comboCount) //得点の計算
    {
        //得点の計算
        chainCount = Mathf.Min(chainCount, ChainPointRatesLength - 1); //連鎖数を更新
        comboCount = Mathf.Min(comboCount, ComboPointRatesLength - 1); //コンボ数を更新
        pointSet.AtomCount = atomCount; //原子の数をセット
        pointSet.FormulaPoint = formulaPoint; //化学式の得点をセット
        pointSet.ChainRate = chainPointRates[chainCount]; //連鎖数に応じた倍率
        pointSet.ComboRate = comboPointRates[comboCount]; //コンボ数に応じた倍率
        pointSet.Point = atomCount * formulaPoint * pointSet.ChainRate * pointSet.ComboRate; //得点を計算
    } 
    public void PassDisturbance(int num, PlayerBase player, Vector3 center)
    {
        if (players[0] == player) players[1].GotDisturbanceNumber(num, center).Forget(); //プレイヤー1のじゃま原子数を設定
        else players[0].GotDisturbanceNumber(num, center).Forget(); //プレイヤー0のじゃま原子数を設定 
    }
    #endregion 公開メソッド

    #region 非公開メソッド
    private void Init() //ゲームの初期化
    {
        cts = new CancellationTokenSource(); //キャンセルトークンを初期化
        gameTime = 0f; //ゲーム時間を初期化
        players = FindObjectsByType<PlayerBase>(FindObjectsSortMode.None); //プレイヤーを取得
        int[] atomCount = new int[(int)AtomType.None]; //原子の数を初期化
        foreach (var formula in Formulas)
        {
            int count = 0; //原子の数を初期化
            foreach (var atom in formula.AtomDictionary)
            {
                count += atom.Value; //総原子の数をカウント
                atomCount[(int)atom.Key] += atom.Value; //原子の数をカウント
            }
        }
        //ボーナス化学式を決める
        //atomCountの数がbonusFormulaの数より大きい化学式から、bonusFormulaをランダムに選択
        var selectedList = Formulas.Where(a => a.AtomCount >= bonusAtomMin).ToArray();
        if (selectedList.Length > 0)
        {
            var bonusIndex = UnityEngine.Random.Range(0, selectedList.Length); // UnityのRandom
            bonusFormula = selectedList[bonusIndex]; //ボーナス化学式を選択
        }
        WeightPerLevel(atomCount); //原子の重みをレベルに応じて設定
        DisturbanceAtomObjects = new List<AtomObject>(20);
        for (int i = 0; i < 20; i++)
        {
            InstantiateDisturbance();
        }
    }
    public void InstantiateDisturbance()
    {
        AtomObject atomObject = Instantiate(atomPrefab, stockAtoms.transform); //原子をインスタンス化
        atomObject.Set(disturbanceAtomColor, disturbanceAtom);
        atomObject.UnEnabled();
        atomObject.transform.localScale *= DisturbancAtomsSize; //おじゃま原子のサイズを変更
        DisturbanceAtomObjects.Add(atomObject);
    }
    private void WeightPerLevel(in int[] atomCount)
    {
        atomWeight = new int[(int)AtomType.None];
        //総合重みを初期化
        fullWeight = 0;
        for (int i = 0; i < (int)AtomType.None; i++)
        {
            if (i == (int)disturbanceAtom) continue; //おじゃま原子の重みを0にする
            //原子の重みをレベルに応じて設定
            if (Level == 0)
            {
                atomWeight[i] = atomCount[i] switch
                {
                    > 60 => 32,
                    > 13 => 16,
                    > 8 => 8,
                    > 3 => 4,
                    > 2 => 2,
                    _ => 1,
                };
            }
            else if(Level == 1)
            {
                atomWeight[i] = atomCount[i] switch
                {
                    > 60 => 16,
                    > 13 => 8,
                    > 8 => 4,
                    > 3 => 2,
                    _ => 1,
                };
            }
            else
            {
                atomWeight[i] = atomCount[i] switch
                {
                    > 60 => 16,
                    > 8 => 8,
                    > 4 => 4,
                    > 2 => 2,
                    _ => 1,
                };
            }
            //総合重みを計算
            fullWeight += atomWeight[i];
        }
    } //原子の重みをレベルに応じて設定するメソッド
    static public Color GetAtomColor(AtomType atomType)
    {
        var atomGroup = AtomGroupHelper.atomGroups[atomType]; //原子のグループを取得
        return groupColors[atomGroup];
    } //原子の色を取得
    public void Pause()
    {
        if (CurrentState == State.Pause)
        {
            CurrentState = State.Play;
        }
        else if(CurrentState == State.Play)
        {
            CurrentState = State.Pause;
        }
    } //ポーズ
    async UniTaskVoid Countdown()
    {
        //カウントダウン処理
        for (int i = 0; i < players.Length; i++)
        {
            players[i].PlayCountdownAsync().Forget(); //カウントダウンのテキストをセット
        }
        await UniTask.WaitForSeconds(3f, ignoreTimeScale: false, cancellationToken: cts.Token); //カウントダウンの待機
        CurrentState = State.Play; //カウントダウン終了後、状態をPlayに変更
        for (int i = 0; i < players.Length; i++)
        {
            players[i].SetStart();
        }
    } //カウントダウンを行うコルーチン   
    void UpdateGameTime()
    {
        // Play状態の処理
        gameTime += Time.deltaTime; //ゲーム時間を更新
        updateDropCount += Time.deltaTime;
        if(updateDropCount >= updateDropTime)
        {
            dropTime *= updateDropTimeRate;
            updateDropCount -= updateDropTime;
        }
    } //ゲーム時間の更新
    #endregion 非公開メソッド

    #region 公開クラス
    public class FormulaObject //化学式を管理するクラス
    {
        public string Name { get; private set; } //化学式の名前
        public string Formula { get; private set; }//化学式
        public Dictionary<AtomType, int> AtomDictionary { get; private set; } //原子の種類とその数を管理する辞書
        public int AtomCount { get; private set; }//原子の数
        public int Point { get; private set; } //得点
        public FormulaObject(string name, string formula, Dictionary<AtomType, int> atomDict)
        {
            Name = name;　//化学式の名前を設定
            Formula = formula;　//化学式を設定
            AtomDictionary = atomDict;　//原子の種類とその数を設定

            AtomCount = atomDict.Values.Sum(); //原子の数をカウント
            Point = atomDict.Sum(pair =>
            {
                var group = AtomGroupHelper.GetGroup(pair.Key);
                var multiplier = group switch
                {
                    AtomGroupType.AlkaliMetal or
                    AtomGroupType.AlkalineEarthMetal or
                    AtomGroupType.TransitionMetal or
                    AtomGroupType.SimpleMetal => 3,
                    _ => 1
                };
                return pair.Value * multiplier;
            });
        }
    }
    public static class AtomGroupHelper
    {
        public static readonly Dictionary<AtomType, AtomGroupType> atomGroups = new Dictionary<AtomType, AtomGroupType>
    {
        { AtomType.H, AtomGroupType.Hydrogen },
        { AtomType.Cl, AtomGroupType.Halogen },
        { AtomType.F,  AtomGroupType.Halogen },
        { AtomType.Br, AtomGroupType.Halogen },
        { AtomType.I,  AtomGroupType.Halogen },
        { AtomType.O,  AtomGroupType.OxygenSulfur },
        { AtomType.S,  AtomGroupType.OxygenSulfur },
        { AtomType.N,  AtomGroupType.NitrogenPhosphorus },
        { AtomType.P,  AtomGroupType.NitrogenPhosphorus },
        { AtomType.C,  AtomGroupType.Carbon },
        { AtomType.Na, AtomGroupType.AlkaliMetal },
        { AtomType.K,  AtomGroupType.AlkaliMetal },
        { AtomType.Mg, AtomGroupType.AlkalineEarthMetal },
        { AtomType.Ca, AtomGroupType.AlkalineEarthMetal },
        { AtomType.Ba, AtomGroupType.AlkalineEarthMetal },
        { AtomType.Fe, AtomGroupType.TransitionMetal },
        { AtomType.Cu, AtomGroupType.TransitionMetal },
        { AtomType.Zn, AtomGroupType.TransitionMetal },
        { AtomType.Ag, AtomGroupType.TransitionMetal },
        { AtomType.Al, AtomGroupType.SimpleMetal },
        { AtomType.None, AtomGroupType.None }
    };

        public static AtomGroupType GetGroup(AtomType atomType)
        {
            return atomGroups.TryGetValue(atomType, out var group) ? group : AtomGroupType.None;
        }
    } //原子を管理するクラス
    public struct PointSet //得点を管理する構造体
    {
        public int Point; //得点
        public int FormulaPoint; //基礎得点
        public int AtomCount; //原子数
        public int ChainRate; //連鎖数
        public int ComboRate; //コンボ数
    }
    #endregion 公開クラス
}
