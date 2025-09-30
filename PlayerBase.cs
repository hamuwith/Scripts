using System.Collections.Generic;
using TMPro;
using UnityEngine;
using System;
using static MainManager;
using System.Threading;
using System.Linq;
using DG.Tweening;
using Cysharp.Threading.Tasks;
using System.Text;
using UnityEngine.Pool;
//未実装は'***'で表記

public class PlayerBase : MonoBehaviour
{
    #region シリアライズフィールド
    [SerializeField] TextMeshProUGUI countdownText; //カウントダウン用のテキスト
    [SerializeField] TextMeshProUGUI formulaText; //化学式テキスト
    [SerializeField] TextMeshProUGUI chemicalText; //化学式テキスト
    [SerializeField] TextMeshProUGUI pointText; //得点テキスト
    [SerializeField] TextMeshProUGUI formulaPointText; //各得点テキスト
    [SerializeField] Vector2 leftBottomPosition ;//左下の位置
    [SerializeField] Vector2[] nextAtomsPosition; //次の原子の位置
    [SerializeField] Vector3 disturbancAtomPosition; //じゃま原子の位置
    #endregion シリアライズフィールド

    #region プライベートフィールド
    Vector3 formulaStart; //化学式表示初期位置
    Vector3 chemicalStart; //化学式表示初期位置
    protected AtomObject[] nextAtoms; //次の原子を管理する配列
    protected PlayState playState; //プレイの状況
    protected AtomObject[] currentAtoms; //現在の原子を管理する配列
    protected Vector2Int[] currentAtomPositions; //現在の原子の位置
    int point; //プレイヤーの得点
    int chainCount; //連鎖数
    int comboCount; //コンボ数
    AtomObject[,] stageAtom; //原子の位置を管理する配列
    protected float[] stageAtomF; //原子番号の位置を管理する配列
    Vector2Int[] dropVector2Ints; //ドロップ表示位置
    protected int[] dropAtomYs; //一番上の原子
    List<Vector2Int> checkBuffer; //探査中原子バッファ
    Dictionary<FormulaObject, HashSet<Vector2Int>> atomObjectHashs; //揃った原子のリスト
    Queue<AtomType> atomsQueue; //キューペア原子
    float dropCount; //落下カウント
    int disturbancStartX; //じゃまスタート位置
    CancellationTokenSource cts; //キャンセルトークン
    SpriteRenderer[] dropPoints; //落下地点オブジェクト
    PointSet pointSet = new PointSet(); //得点を管理する
    ObjectPool<AtomObject> atomObjectPool; //原子のプール
    StringBuilder stringBuilder; //テキスト
    Vector2Int[] vector2Ints; //移動量
    int passTotalDisturbanceNumber; //渡したじゃま原子数
    protected int gotDisturbanceNumber; //取得したじゃま原子数
    #endregion プライベートフィールド

    #region プロテクトフィールド
    /// <summary> メインマネージャー </summary>
    protected MainManager M { get; private set; } //メインマネージャー
    #endregion プロテクトフィールド

    #region 定数
    const int displayUpper = 12; //ステージの表示最大高さ
    readonly Color countdownTextColor = Color.white; //カウントダウンテキストカラー
    #endregion 定数

    #region プロパティ    
    #endregion プロパティ

    #region 列挙体
    protected enum PlayState //プレイ遷移
    {
        None,
        Move,
        NoMove,
        GameOver,
    }
    public enum Direction //方向
    {
        Right,
        Left,
        Down,
        None,
    }
    public enum RotationInfo //回転情報
    {
        Top,
        Right,
        Down,
        Left,
    }
    #endregion 列挙体

    #region 公開メソッド
    public virtual void Start0(int id) //プレイヤーの初期化を行うメソッド
    {
        M = Instance; //MainManagerのインスタンスを取得
        //ゲームの初期化
        Init();
    }
    public void ReadyAtom() //原子の準備
    {
        //次の原子までをセット
        if (nextAtoms[0] == null)
        {
            SetAtoms(false); //原子をセット
        }
    }
    public void UpdatePlayState() //Play状態の処理を行うメソッド
    {
        if (playState == PlayState.Move) PlayMove();
    }
    public AtomObject SetAtom(AtomType atomType, PlayerBase player) //現在原子をセット
    {
        //AtomObject atomObject = atomObjects.First(x => x.IsExist == false);
        AtomObject atomObject = atomObjectPool.Get();
        var atomGroup = AtomGroupHelper.atomGroups[atomType]; //原子のグループを取得
        atomObject.Set(GetAtomColor(atomType), atomType);
        atomObject.transform.parent = player.transform; //親をプレイヤーに設定
        return atomObject;
    }
    public void SetPoint(int atomCount, int formulaPoint, int chainCount, int comboCount) //得点の計算
    {
        M.CalcPoint(ref pointSet, atomCount, formulaPoint, chainCount, comboCount);
        point += pointSet.Point;
    }
    public async UniTaskVoid GotDisturbanceNumber(int num, Vector3 center) //じゃま原子をセットするメソッド
    {
        Vector3 vector3 = disturbancAtomPosition;
        Tween tween = null;
        var startIndex = gotDisturbanceNumber;
        gotDisturbanceNumber += num;
        var endIndex = gotDisturbanceNumber;
        if (startIndex > 0)
        {
            vector3.x = M.DisturbanceAtomObjects[startIndex - 1].transform.localPosition.x + M.DisturbancAtomsSize;
        }
        for (int i = startIndex; i < endIndex; i++)
        {
            if (M.DisturbanceAtomObjects.Count <= i) M.InstantiateDisturbance();
            M.DisturbanceAtomObjects[i].Enabled(); //じゃま原子を追加
            M.DisturbanceAtomObjects[i].transform.parent = transform; //親をプレイヤーに設定
            M.DisturbanceAtomObjects[i].transform.position = center;
            tween = M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            vector3.x += M.DisturbancAtomsSize;
        }
        await tween?.AsyncWaitForCompletion();
        if (NormalMax < gotDisturbanceNumber)
        {
            vector3.x = M.DisturbanceAtomObjects[0].transform.localPosition.x;
            for (int i = 1; i < gotDisturbanceNumber; i++)
            {
                vector3.x += (M.DisturbancAtomsSize * (NormalMax - 1)) / (gotDisturbanceNumber - 1);
                M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            }
        }
    }
    #endregion 公開メソッド

    #region 非公開メソッド     
    protected void Init() //プレイヤーの初期化
    {
        vector2Ints = new Vector2Int[2]; //移動量を初期化
        dropVector2Ints = new Vector2Int[2]; //ドロップ地点を初期化
        cts = new CancellationTokenSource(); //キャンセルトークンを初期化
        M.onCreateAtomType += type => atomsQueue.Enqueue(type); //原子のキューを作成
        atomObjectHashs = new Dictionary<FormulaObject, HashSet<Vector2Int>>(); //揃った原子のリストを初期化
        atomsQueue = new Queue<AtomType>(); //ペア原子のキューを初期化
        stageAtom = new AtomObject[M.Size.x, M.Size.y]; //原子の位置を初期化
        stageAtomF = new float[M.Size.x * M.Size.y]; //原子番号の位置を初期化
        dropAtomYs = new int[M.Size.x];
        nextAtoms = new AtomObject[4]; //次の原子を初期化
        currentAtoms = new AtomObject[2]; //現在の原子を初期化
        currentAtomPositions = new Vector2Int[2];
        checkBuffer = new List<Vector2Int>(23); //探査中原子バッファを初期化
        //ドロップ地点を初期化
        dropPoints = new SpriteRenderer[2];
        for(int i = 0; i< dropPoints.Length; i++)        
        {
            dropPoints[i] = Instantiate(M.DropPointPrefab, transform); //ドロップ地点を生成
            dropPoints[i].enabled = false; //ドロップ地点を非表示にする
        }
        atomObjectPool = new ObjectPool<AtomObject>(
           createFunc: () => Instantiate(M.AtomPrefab),
           actionOnGet:go =>
           {
               go.Enabled();
               go.transform.parent = transform; //親をプレイヤーに設定
           },
           actionOnRelease: go => 
           {
               go.UnEnabled();
               go.transform.parent = M.StockAtoms.transform; //親をnullに設定
           },
           collectionCheck: false,
           defaultCapacity: 100,
           maxSize: M.StageSize + 10
        );
        formulaPointText.enabled = false;
        formulaText.enabled = false;
        chemicalText.enabled = false;
        formulaStart = formulaText.rectTransform.position;
        chemicalStart = chemicalText.rectTransform.position;
        pointText.text = $"{point.ToString()}P" ;
        stringBuilder = new StringBuilder();
    }
    protected virtual void SetAtoms(bool isStart = true) //原子をセットするメソッド
    {
        //原子を移動
        for (int i = 0; i < 2; i++)
        {
            //現在の原子の設定
            currentAtoms[i] = nextAtoms[i]; //次の原子を現在の原子に設定
            if (currentAtoms[i] != null)
            {
                var vector = new Vector2Int(M.StartPosition.x, M.StartPosition.y + i);
                Set(currentAtoms[i], vector, i); //原子を移動
            }
            //次の原子の設定
            nextAtoms[i] = nextAtoms[i + 2]; //次の次の原子を次の原子に設定
            if (nextAtoms[i] != null)
            {
                nextAtoms[i].transform.localPosition = new Vector3(nextAtomsPosition[0].x, nextAtomsPosition[0].y + i, 0f); //原子の位置を設定
            }
            //次の次の原子の設定
            if (atomsQueue.Count <= 0)
            {
                M.CreateAtoms(); //ペア原子を生成
            }
            nextAtoms[i + 2] = SetAtom(atomsQueue.Dequeue(), this); //原子をセット
            nextAtoms[i + 2].transform.localPosition = new Vector3(nextAtomsPosition[1].x, nextAtomsPosition[1].y + i, 0f); ;
        }
        if (isStart)
        {
            playState = PlayState.Move;
            SetDropPoint(ref dropVector2Ints, ref currentAtomPositions);
            //ドロップカウントを半分にする
            dropCount = M.DropTime / 2;
        }
    }  
    private void Set(AtomObject atomObject, in Vector2Int currect, int index) //原子をセット
    {
        if (stageAtom[currect.x, currect.y] != null)
        {
            playState = PlayState.GameOver;
        }
        //原子を移動
        atomObject.transform.localPosition = currect + leftBottomPosition; //Transformの位置を設定
        SettingDisplay(currect.y, atomObject);
        currentAtomPositions[index] = currect;
    }    
    void SetFreeAtom(int x, int y, AtomObject atomObject) //原子を設定するメソッド
    {
        var vector2Int = new Vector2Int(x, y);
        stageAtom[x, y] = atomObject; //原子の位置を設定
        stageAtomF[x + y * M.Size.x] = (int)atomObject.AtomType + 1; //原子番号を設定
        stageAtom[x, y].transform.localPosition = vector2Int + leftBottomPosition; //Transformの位置を設定
    }
    protected virtual void PlayMove() //移動入力を処理するメソッド
    {
#if DEBUG
        DropStep(Time.deltaTime * 0.3f); //ドロップ処理
#else
        DropStep(Time.deltaTime); //ドロップ処理
#endif
    }
    protected void DropStep(float deltaTime) //移動入力を処理するメソッド
    {
        //ドロップカウントを更新
        dropCount += deltaTime;
        //ドロップカウントが一定時間を超えたら下移動
        if (dropCount >= M.DropTime)
        {
            bool move = Move(Direction.Down);
            dropCount = 0; //ドロップカウントをリセット
            if (!move)
            {
                foreach(var dropPoint in dropPoints)
                {
                    dropPoint.enabled = false; //ドロップ地点を非表示にする
                }
                EndCheck().Forget();
            }
        }
    }
    async UniTaskVoid EndCheck() //チェックかすべてからか判定
    {
        playState = PlayState.NoMove;
        await FreeFall();
        await CheckOrFullClear();
        var isEnd = !DropGotDisturbanceNumber();
        if (!isEnd)
        {
            await FreeFall(true);
            await CheckOrFullClear();
        }
        SetAtoms();
    }
    async UniTask CheckOrFullClear() //チェックかすべてからか判定
    {
        chainCount = 0;
        comboCount = 0;
        while (true)
        {
            if (Check())
            {
                //化学式をチェックする
                var center = await PointCount(); //得点を計算
                await AtomDelete(); //原子を削除
                await FreeFall(true);
                SetDisturbance(center);
            }
            else
            {
                break;
            }
            bool allNull = true;
            //一番下がnullかどうか
            for (int i = 0; i < M.Size.x; i++)
            {
                if (stageAtom[i, 0] != null)
                {
                    allNull = false;
                    break;
                }
            }
            if (allNull)
            {
                //すべての原子がない
                await FullClear();
                break;
            }
        }
    }
    public async UniTaskVoid PlayCountdownAsync() //カウントダウンメソッド
    {
        string[] texts = { "3", "2", "1", "Go!" };
        foreach (var text in texts)
        {
            var end = text == "Go!";
            countdownText.text = text;
            countdownText.color = countdownTextColor;
            countdownText.transform.localScale = Vector3.zero;
            var sequence = DOTween.Sequence();
            sequence.Append(countdownText.transform.DOScale(1.5f, 0.2f).SetEase(end ? Ease.OutSine : Ease.OutBack));
            sequence.Join(countdownText.DOFade(1f, 0.1f));
            if(!end) sequence.AppendInterval(0.5f);
            sequence.Append(countdownText.DOFade(0f, end ? 1.4f : 0.3f).SetEase(end ? Ease.InOutSine : Ease.Unset));
            sequence.Join(countdownText.transform.DOScale(end ? 2.4f : 2f, end ? 1.4f : 0.3f).SetEase(end ? Ease.InOutSine : Ease.Unset));
            await sequence.AsyncWaitForCompletion();
        }
        countdownText.gameObject.SetActive(false);
    }
    public virtual void SetStart() //ゲーム開始
    {
        SetAtoms(); //原子をセット
    }
    bool Check() //化学式をチェックするメソッド
    {
        bool ok = false;
        foreach (var formula in M.Formulas)
        {
            ok |= CheckConnection(formula);
        }
        return ok;
    }
    async UniTask<Vector3> PointCount() //得点を計算、表示
    {
        Vector3 center = new Vector3();
        //得点の計算
        var sortedFormulas = atomObjectHashs
            .Select(kv => kv)
            .ToList();
        sortedFormulas.Sort((a, b) => (a.Key.Point * a.Value.Count).CompareTo(b.Key.Point * b.Value.Count));
        for (int i = 0; i < sortedFormulas.Count; i++)
        {
            int prePoint = point;
            SetPoint(sortedFormulas[i].Value.Count, sortedFormulas[i].Key.Point, chainCount, comboCount); //得点を計算
            chainCount++; //連鎖数を更新
            FormulaText(sortedFormulas[i].Key.Name).Forget();
            ChemicalText(sortedFormulas[i].Key.Formula).Forget();
            DisplayText(sortedFormulas[i].Value, prePoint).Forget();
            ScaleAtom(sortedFormulas[i].Value, out center);
            await UniTask.WaitForSeconds(1f, cancellationToken: cts.Token);
        }    
        comboCount++; //コンボ数を更新
        chainCount = 0;
        return center;
    }
    void ScaleAtom(HashSet<Vector2Int> vector2Ints, out Vector3 center) //揃った化学式を拡大表示する
    {
        Vector3 vectorAtom = new Vector3();
        center = new Vector3();
        center.x = (vector2Ints.Max(p => p.x) + vector2Ints.Min(p => p.x)) / 2f + leftBottomPosition.x;
        center.y = (vector2Ints.Max(p => p.y) + vector2Ints.Min(p => p.y)) / 2f + leftBottomPosition.y;
        foreach (var vector2Int in vector2Ints)
        {
            AtomObject atomObject = stageAtom[vector2Int.x, vector2Int.y];
            atomObject.spriteRenderer.sortingOrder = 20;
            float scale = atomObject.transform.localScale.x;
            vectorAtom = (atomObject.transform.localPosition - center) * 0.5f + atomObject.transform.localPosition;
            var seq = DOTween.Sequence();
            seq.Append(atomObject.transform.DOScale(scale * 1.5f, 0.2f))
                .Join(atomObject.transform.DOLocalMove(vectorAtom, 0.2f))
                .AppendInterval(0.4f)
                .Append(atomObject.transform.DOScale(scale, 0.2f))
                .Join(atomObject.transform.DOLocalMove(atomObject.transform.localPosition, 0.2f))
                .OnComplete(() => atomObject.spriteRenderer.sortingOrder = 0);
        }
        center.x += transform.position.x;
        center.y += transform.position.x;
    }
    async UniTaskVoid DisplayText(HashSet<Vector2Int> vector2Ints, int prePoint)
    {
        await FormulaPointAsync(vector2Ints);
        PointAsync(prePoint);
    }
    async UniTask FormulaPointAsync(HashSet<Vector2Int> vector2Ints)//化学式得点表示
    {
        stringBuilder.Append($"{pointSet.AtomCount.ToString()}x{pointSet.FormulaPoint.ToString()}");
        if (pointSet.ChainRate != 1) stringBuilder.Append($"x{pointSet.ChainRate.ToString()}");
        if (pointSet.ComboRate != 1) stringBuilder.Append($"x{pointSet.ComboRate.ToString()}");
        formulaPointText.text = stringBuilder.ToString(); //得点を表示
        stringBuilder.Clear();
        formulaPointText.enabled = true;
#if DEBUG
        await UniTask.WaitForSeconds(0.8f, cancellationToken: cts.Token);
#else
        vector3 = transform.position;
        vector3.x += (float)(vector2Ints.Average(p => p.x) + leftBottomPosition.x);
        vector3.y += (float)(vector2Ints.Average(p => p.y) + leftBottomPosition.y);
        formulaPointText.rectTransform.position = vector3; //化学式テキストの位置を設定
        formulaPointText.rectTransform.localScale = vector3One;
        vector3.y += 0.5f;
        var seq = DOTween.Sequence();
        seq.Append(formulaPointText.rectTransform.DOMove(vector3, 0.5f).SetEase(Ease.OutCubic))
            .Join(formulaPointText.DOFade(1f, 0.1f))
            .Join(formulaPointText.rectTransform.DOScale(1.6f, 0.3f))
            .Append(formulaPointText.rectTransform.DOMove(pointText.rectTransform.position, 0.3f).SetEase(Ease.InCubic))
            .Join(DOTween.Sequence()
                .AppendInterval(0.2f)
                .Append(formulaPointText.DOFade(0f, 0.1f)));
        await seq.AsyncWaitForCompletion(); //表示時間
#endif
        formulaPointText.enabled = false;
    }
    void PointAsync(int prePoint)//得点表示
    {
        DOVirtual.Int(prePoint, point, 0.7f, value =>
        {
            pointText.text = $"{value.ToString("N0")}P";
        });
        var scaleSeq = DOTween.Sequence();
        scaleSeq.Append(pointText.rectTransform.DOScale(1.2f, 0.1f))
            .AppendInterval(0.6f)
            .Append(pointText.rectTransform.DOScale(1f, 0.1f));
    }
    async UniTask AtomDelete() //原子の削除
    {
        //原子の削除
        foreach (var atomObjectHash in atomObjectHashs)
        {
            //揃った原子を削除
            foreach (var atom in atomObjectHash.Value)
            {
                if (stageAtom[atom.x, atom.y] != null)
                {
                    atomObjectPool.Release(stageAtom[atom.x, atom.y]); //原子をプールに戻す
                    stageAtom[atom.x, atom.y] = null;
                    stageAtomF[atom.x + atom.y * M.Size.x] = 0; //原子番号を設定
                    if (dropAtomYs[atom.x] > atom.y) dropAtomYs[atom.x] = atom.y;
                }
            }
        }
        //揃った原子を削除
        atomObjectHashs.Clear(); 
        await UniTask.WaitForSeconds(1.0f, ignoreTimeScale: false, cancellationToken: cts.Token); //表示時間
    }
    async UniTaskVoid FormulaText(string formulaName) //化学式表示
    {
        formulaText.rectTransform.position = formulaStart;
        formulaText.enabled = true;
        formulaText.rectTransform.localScale = Vector3.one;
        formulaText.text = formulaName;
        var seq = DOTween.Sequence();
        seq.Append(formulaText.rectTransform.DOScale(1.5f, 0.6f))
            .Join(DOTween.Sequence()
                .Join(formulaText.DOFade(1f, 0.1f))
                .AppendInterval(0.4f)
                .Append(formulaText.DOFade(0f, 0.4f)));
        await seq.AsyncWaitForCompletion(); //表示時間
        formulaText.enabled = false;
    }
    async UniTaskVoid ChemicalText(string formulaName) //化学式表示
    {
        chemicalText.rectTransform.position = chemicalStart;
        chemicalText.enabled = true;
        chemicalText.rectTransform.localScale = Vector3.one;
        chemicalText.text = formulaName;
        var seq = DOTween.Sequence();
        seq.Append(chemicalText.rectTransform.DOScale(1.5f, 0.6f))
            .Join(DOTween.Sequence()
                .Join(chemicalText.DOFade(1f, 0.1f))
                .AppendInterval (0.4f)
                .Append(chemicalText.DOFade(0f, 0.4f)));
        await seq.AsyncWaitForCompletion(); //表示時間
        chemicalText.enabled = false;
    }
    async UniTask FullClear() //ステージが空のとき
    {
        FormulaText("全消し").Forget();
        for (int i = 0; i < M.Size.x; i++)
        {
            AtomObject atomObject = atomObjectPool.Get();
            var scale = atomObject.transform.localScale.x;
            atomObject.Set(GetAtomColor(M.FullClearAtom), M.FullClearAtom);
            SetFreeAtom(i, 0, atomObject); //原子の位置を設定
            dropAtomYs[i] = 1;
            atomObject.transform.localScale = Vector3.zero;
            atomObject.transform.DOScale(scale, 1f).SetEase(Ease.OutBounce);
        }
        await UniTask.WaitForSeconds(1.0f, ignoreTimeScale: false, cancellationToken: cts.Token); //表示時間
    }
    protected bool Move(Direction direction) //移動メソッド
    {
        //移動できるかどうか
        bool isSet = false;
        //移動量を設定
        if (direction == Direction.Right)
        {
            for (int i = 0; i < vector2Ints.Length; i++)
            {
                vector2Ints[i].x = 1;
                vector2Ints[i].y = 0;
            }
        }
        else if (direction == Direction.Left)
        {
            for (int i = 0; i < vector2Ints.Length; i++)
            {
                vector2Ints[i].x = -1;
                vector2Ints[i].y = 0;
            }

        }
        else if (direction == Direction.Down)
        {
            for (int i = 0; i < vector2Ints.Length; i++)
            {
                vector2Ints[i].x = 0;
                vector2Ints[i].y = -1;
            }
        }
        //移動可能なら移動
        if (isSet = CheckMovement(vector2Ints))
        {
            SettingPair(vector2Ints);
        }
        return isSet;
    }
    async UniTask FreeFall(bool all = false) //原子を自由落下させるメソッド
    {
        List<UniTask> uniTasks = new List<UniTask>();
        if (all)
        {
            //すべての原子を自由落下させる
            for (int i = 0; i < M.Size.x; i++)
            {
                bool isBound = true;
                for (int j = dropAtomYs[i] + 1; j < M.Size.y; j++)
                {
                    if (stageAtom[i, j] != null)
                    {
                        uniTasks.Add(FreeFalling(i, j, isBound));
                        isBound = false;
                    }
                    else
                    {
                        isBound = true;
                    }
                }
            }
        }
        else
        {
            //現在の原子をステージに設定
            stageAtom[currentAtomPositions[0].x, currentAtomPositions[0].y] = currentAtoms[0];
            stageAtom[currentAtomPositions[1].x, currentAtomPositions[1].y] = currentAtoms[1];
            stageAtomF[currentAtomPositions[0].x + currentAtomPositions[0].y * M.Size.x] = (int)currentAtoms[0].AtomType + 1; //原子番号を設定
            stageAtomF[currentAtomPositions[1].x + currentAtomPositions[1].y * M.Size.x] = (int)currentAtoms[1].AtomType + 1; //原子番号を設定
            //現在の原子を自由落下させる
            int underIndex = currentAtomPositions[0].y < currentAtomPositions[1].y ? 0 : 1;
            bool isBound = currentAtomPositions[0].x != currentAtomPositions[1].x;
            uniTasks.Add(FreeFalling(currentAtomPositions[underIndex].x, currentAtomPositions[underIndex].y));
            uniTasks.Add(FreeFalling(currentAtomPositions[1 - underIndex].x, currentAtomPositions[1 - underIndex].y, isBound));
            foreach(var dropPoint in dropPoints)
            {
                dropPoint.enabled = false; //ドロップ地点を非表示にする
            }
        }
        await UniTask.WhenAll(uniTasks);
    }
    public async void Drop() //原子をドロップするメソッド
    {
        if (M.CurrentState != State.Play) return;
        if (playState == PlayState.NoMove) return;
        //ドロップ可能かどうか
        int dropNum = M.Size.y;
        //ドロップ最大位置
        for(int i = 0; i < currentAtomPositions.Length; i++)
        {
            dropNum = Mathf.Min(dropNum, currentAtomPositions[i].y - dropAtomYs[currentAtomPositions[i].x]);
        }
        if (dropNum != 0)
        {
            UniTask[] uniTasks = new UniTask[currentAtoms.Length];
            var duration = dropNum * 0.06f + 0.1f;
            for (int i = 0; i < currentAtoms.Length; i++)
            {
                currentAtomPositions[i].y -= dropNum;
                var isBound = currentAtomPositions[i].y - 1 <= 0 || stageAtom[currentAtomPositions[i].x, currentAtomPositions[i].y - 1] != null;
                uniTasks[i] = DoFreeFalling(currentAtoms[i], dropNum, currentAtomPositions[i].y, isBound);
            }
            SetDropPoint(ref dropVector2Ints, ref currentAtomPositions);
            //ドロップ位置を設定
            playState = PlayState.NoMove;
            await UniTask.WhenAll(uniTasks);
        }
        //状態の変更
        EndCheck().Forget();
    }
    bool CheckConnection(FormulaObject formula) //盤面のAtomObjectをFormulaObjectと比較し、つながりを判定するメソッド
    {
        bool existed = false;

        // 盤面のサイズを取得
        int width = M.Size.x;
        int height = M.Size.y;

        //// 探索済みフラグ
        bool[] visited = new bool[width * height];

        // 盤面を走査
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                if (stageAtom[x, y] != null && (!existed || (existed && !atomObjectHashs[formula].Contains(new Vector2Int(x, y)))))
                {
                    // FormulaObjectの構成要素を取得
                    Dictionary<AtomType, int> requiredAtoms = new Dictionary<AtomType, int>(formula.AtomDictionary);
                    checkBuffer.Clear(); // 探査中原子バッファをクリア
                    visited.AsSpan().Fill(false);
                    // つながりを探索
                    if (ExploreConnections(x, y, visited, requiredAtoms, formula.AtomCount, width, height))
                    {
                        // すべての必要な原子が見つかった場合
                        //atomObjectHashs[formula].UnionWith(checkBuffer); // 揃った原子を保存
                        atomObjectHashs.TryAdd(formula, new HashSet<Vector2Int>());
                        atomObjectHashs[formula].UnionWith(checkBuffer);
                        existed = true;
                    }
                }
            }
        }
        return existed; // つながりが成立しない
    }
    bool ExploreConnections(int startX, int startY, bool[] visited, Dictionary<AtomType, int> requiredAtoms, int atomCount, int width, int height) //必要な範囲内で隣接するAtomObjectを探索するメソッド
    {
        Stack<Vector2Int> stack = new Stack<Vector2Int>();
        stack.Push(new Vector2Int(startX, startY));
        var vector2Int = new Vector2Int(); // 探索中原子バッファ
        while (stack.Count > 0)
        {
            Vector2Int current = stack.Pop();
            int x = current.x;
            int y = current.y;
            // 探索範囲外の場合は次へ
            if (x < 0 || x >= width || y < 0 || y >= height) continue;

            // すでに探索済み、または空の場合は次へ
            if (visited[GetIndex(x, y, width)] || stageAtom[x, y] == null) continue;

            // 現在の位置を探索済みに設定
            visited[GetIndex(x, y, width)] = true;

            // 現在のAtomObjectを取得
            AtomObject atom = stageAtom[x, y];

            // 使用しないまたは必要数を超えた場合は探索を次へ
            if (!requiredAtoms.ContainsKey(atom.AtomType) || requiredAtoms[atom.AtomType] <= 0)
            {
                continue;
            }
            // 必要な原子をカウント
            requiredAtoms[atom.AtomType]--;
            atomCount--;
            // 揃った原子を保存
            vector2Int.x = x;
            vector2Int.y = y;
            checkBuffer.Add(vector2Int);

            // すべての必要な原子が見つかった場合は終了
            if (atomCount <= 0)
            {
                return true;
            }

            // 隣接する位置をスタックに追加
            foreach (var (dx, dy) in new[] { (1, 0), (-1, 0), (0, 1), (0, -1) })
            {
                stack.Push(new Vector2Int(x + dx, y + dy));
            }
        }
        return false; // つながりが成立しない

        int GetIndex(int x, int y, int width) => y * width + x;
    }
    private bool CheckMovement(in Vector2Int[] position) //移動のチェック
    {
        int[] targetX = new int[position.Length];
        int[] targetY = new int[position.Length];
        bool ok = true; //移動可能かどうか
        for (int i = 0; i < targetX.Length; i++)
        {
            targetX[i] = currentAtomPositions[i].x + position[i].x;
            targetY[i] = currentAtomPositions[i].y + position[i].y;
            ok &= 0 <= targetX[i] && targetX[i] < M.Size.x && 0 <= targetY[i] && targetY[i] < M.Size.y && stageAtom[targetX[i], targetY[i]] == null;
        }
        return ok;
    }
    private void SettingPair(in Vector2Int[] positions) //原子をセット
    {
        //移動
        for (int i = 0; i < currentAtomPositions.Length; i++)
        {
            currentAtomPositions[i] += positions[i];
            currentAtoms[i].transform.localPosition = currentAtomPositions[i] + leftBottomPosition; //Transformの位置を設定
            SettingDisplay(currentAtomPositions[i].y, currentAtoms[i]);
        }
        SetDropPoint(ref dropVector2Ints, ref currentAtomPositions);
    }
    private void SetDropPoint(ref Vector2Int[] currents, ref Vector2Int[] currentAtomPositions) //ドロップ位置設定
    {
        if(currentAtomPositions[0].y == currentAtomPositions[1].y)
        {
            for (int i = 0; i < currentAtomPositions.Length; i++)
            {
                currents[i].x = currentAtomPositions[i].x;
                currents[i].y = dropAtomYs[currentAtomPositions[i].x];
                if (currentAtomPositions[i].y <= currents[i].y)
                {
                    dropPoints[i].enabled = false; //ドロップ地点を非表示にする
                }
                else
                {
                    if(dropPoints[i].enabled == false) dropPoints[i].enabled = true; //ドロップ地点を表示にする
                    dropPoints[i].transform.localPosition = currents[i] + leftBottomPosition; //ドロップ地点の位置を設定
                }
            }
        }
        else
        {
            int lowerY = Mathf.Min(currentAtomPositions[0].y, currentAtomPositions[1].y);
            for (int i = 0; i < currentAtomPositions.Length; i++)
            {
                currents[i].x = currentAtomPositions[i].x;
                currents[i].y = dropAtomYs[currentAtomPositions[i].x] + i;
                if (lowerY <= currents[i].y)
                {
                    dropPoints[i].enabled = false; //ドロップ地点を非表示にする
                }
                else
                {
                    if (dropPoints[i].enabled == false) dropPoints[i].enabled = true; //ドロップ地点を表示にする
                    dropPoints[i].transform.localPosition = currents[i] + leftBottomPosition; //ドロップ地点の位置を設定
                }
            }
        }
    }
    private void SettingDisplay(int y, AtomObject atomObject) //原子をセット
    {
        if (y >= displayUpper)
        {
            atomObject.UnEnabled();
        }
        else
        {
            atomObject.Enabled();
        }
    }
    protected void Rotation(Direction direction) //原子を回転させるメソッド
    {
        if (M.CurrentState != MainManager.State.Play) return;
        //回転可能かどうか
        bool isSet = false;
        //回転情報を取得
        RotationInfo rotationInfo = RotationInfo.Top;
        if (currentAtomPositions[0].x < currentAtomPositions[1].x) rotationInfo = RotationInfo.Right;
        else if (currentAtomPositions[0].x > currentAtomPositions[1].x) rotationInfo = RotationInfo.Left;
        else if (currentAtomPositions[0].y > currentAtomPositions[1].y) rotationInfo = RotationInfo.Down;
        //回転位置を設定
        SetRotationPosition(ref vector2Ints, direction, rotationInfo);
        //回転可能なら回転
        if (isSet = CheckMovement(vector2Ints))
        {
            SettingPair(vector2Ints);
        }
        else
        {
            SetRotationPosition2(ref vector2Ints, direction, rotationInfo);
            if (isSet = CheckMovement(vector2Ints))
            {
                SettingPair(vector2Ints);
            }
        }
    }
    void SetRotationPosition(ref Vector2Int[] vector2Ints, Direction direction, RotationInfo rotationInfo) //回転位置をセット
    {
        //軸はまわらない
        vector2Ints[0].x = 0;
        vector2Ints[0].y = 0;
        //回転位置を設定
        if (direction == Direction.Left)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = -1;
            }
            else if (rotationInfo == RotationInfo.Right)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = -1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = 1;
            }
            else
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = 1;
            }
        }
        else if (direction == Direction.Right)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = -1;
            }
            else if (rotationInfo == RotationInfo.Right)
            {
                vector2Ints[1].x = -1;
                vector2Ints[1].y = 1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = 1;
            }
            else
            {
                vector2Ints[1].x = 1;
                vector2Ints[1].y = -1;
            }
        }
    }
    void SetRotationPosition2(ref Vector2Int[] vector2Ints, Direction direction, RotationInfo rotationInfo) //回転位置をセット
    {
        //回転位置を設定
        if (direction == Direction.Left)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[0].x -= 1;
                vector2Ints[1].x -= 1;
            }
            else if (rotationInfo == RotationInfo.Right)
            {
                vector2Ints[0].y += 1;
                vector2Ints[1].y += 1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[0].x += 1;
                vector2Ints[1].x += 1;
            }
        }
        else if (direction == Direction.Right)
        {
            if (rotationInfo == RotationInfo.Top)
            {
                vector2Ints[0].x += 1;
                vector2Ints[1].x += 1;
            }
            else if (rotationInfo == RotationInfo.Down)
            {
                vector2Ints[0].x -= 1;
                vector2Ints[1].x -= 1;
            }
            else if (rotationInfo == RotationInfo.Left)
            {
                vector2Ints[0].y += 1;
                vector2Ints[1].y += 1;
            }
        }
    }
    async UniTask FreeFalling(int x, int y, bool isBound = true) //原子を自由落下させるメソッド
    {
        if (dropAtomYs[x] < y)
        {
            //落下位置
            AtomObject atomObject = stageAtom[x, y]; //原子を取得
            stageAtom[x, dropAtomYs[x]] = atomObject; //元の原子を保存
            stageAtomF[x + dropAtomYs[x] * M.Size.x] = (int)atomObject.AtomType + 1; //原子番号を設定
            stageAtom[x, y] = null; //元の位置をnullにする
            stageAtomF[x + y * M.Size.x] = 0; //原子番号を設定
            var height = y - dropAtomYs[x];
            dropAtomYs[x]++;
            //原子を移動
            await DoFreeFalling(atomObject, height, dropAtomYs[x] - 1, isBound);
        } 
        else
        {
            dropAtomYs[x]++;
            SettingDisplay(y, stageAtom[x, y]);
        }
    }
    async UniTask DoFreeFalling(AtomObject atomObject, int height, int y, bool isBound)
    {
        var tween = atomObject.transform.DOLocalMoveY(y + leftBottomPosition.y, height * 0.02f)
            .SetEase(Ease.Linear)
            .OnUpdate(() => 
            {
                if (atomObject.transform.localPosition.y >= displayUpper + leftBottomPosition.y)
                {
                    atomObject.UnEnabled();
                }
                else
                {
                    atomObject.Enabled();
                }
            });
        await tween.AsyncWaitForCompletion(); //移動時間
        var scale = atomObject.transform.localScale;
        scale.y *= isBound ? 0.87f : 0.91f;
        scale.x *= isBound ? 1.13f : 1.09f;
        tween = atomObject.transform.DOScale(scale, 0.15f)
            .SetLoops(2, LoopType.Yoyo)
            .SetEase(Ease.InOutSine);
        await tween.AsyncWaitForCompletion();
    }
    int DestroyGotDisturbanceNumber(int num) //じゃま原子を削除するメソッド
    {
        Vector3 vector3 = disturbancAtomPosition;
        var pass = num - gotDisturbanceNumber;
        var start = gotDisturbanceNumber;
        gotDisturbanceNumber = Mathf.Max(gotDisturbanceNumber - num, 0);
        for (int i = start - 1; i >= gotDisturbanceNumber; i--)
        {
            M.DisturbanceAtomObjects[i].UnEnabled(); //じゃま原子を削除
            M.DisturbanceAtomObjects[i].transform.parent = M.StockAtoms.transform;
        }
        if(start <= NormalMax) return pass;
        if (NormalMax < gotDisturbanceNumber)
        {
            for (int i = 1; i < gotDisturbanceNumber; i++)
            {
                vector3.x += (M.DisturbancAtomsSize * (NormalMax - 1)) / (gotDisturbanceNumber - 1);
                M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            }
        }
        else
        {
            for (int i = 1; i < gotDisturbanceNumber; i++)
            {
                vector3.x += M.DisturbancAtomsSize;
                M.DisturbanceAtomObjects[i].transform.DOLocalMove(vector3, 0.5f);
            }
        }
        return pass;
    }
    bool DropGotDisturbanceNumber() //じゃま原子を落とすメソッド
    {
        if (gotDisturbanceNumber <= 0)
        {
            return false;
        }
        int width = M.Size.x;
        int height = M.Size.y;
        int count = 0;
        bool[] isSet = new bool[width];
        for(int i = 0; i < gotDisturbanceNumber; i++)
        {
            M.DisturbanceAtomObjects[i].UnEnabled();
            M.DisturbanceAtomObjects[i].transform.parent = M.StockAtoms.transform;
        }
        for (int y = height - 1; y >= 0; y--)
        {
            for (int i = 0; i < width; i++)
            {
                var x = (disturbancStartX + i) % width;
                if (isSet[x]) continue;
                if (stageAtom[x, y] == null)
                {
                    var disturbanceAtomObject = atomObjectPool.Get();
                    disturbanceAtomObject.Set(M.DisturbanceAtomColor, M.DisturbanceAtom);
                    SetFreeAtom(x, y, disturbanceAtomObject); //原子の位置を設定
                    count++;
                    if(count >= gotDisturbanceNumber)
                    {
                        disturbancStartX = (x + 1) % width;
                        gotDisturbanceNumber = 0;
                        return true;
                    }
                }
                else
                {
                    isSet[x] = true; //埋まっている
                    //すべて埋まっていたら
                    var all = isSet.All(p => p);
                    if (all)
                    {
                        disturbancStartX = (x + 1) % width;
                        gotDisturbanceNumber = 0;
                        return true;
                    }
                }
            }
        }
        gotDisturbanceNumber = 0;
        return true;
    }
    void SetDisturbance(Vector3 center) //おじゃま原子をセットするメソッド
    {
        var disturbanceCount = point / 400 - passTotalDisturbanceNumber; //追加じゃま原子
        passTotalDisturbanceNumber = point / 400; //更新
        if (disturbanceCount > 0)
        {
            var disturbanceNumber = DestroyGotDisturbanceNumber(disturbanceCount); //じゃま原子を削除
            if (disturbanceNumber > 0) M.PassDisturbance(disturbanceNumber, this, center); //じゃま原子の数を渡す
        }
    }
    private void OnDestroy() //削除時、タスクをキャンセル
    {
        //キャンセルトークンをキャンセル
        if (cts != null)
        {
            cts.Cancel();
            cts.Dispose();
            cts = null;
        }
    }
#endregion 非公開メソッド
}