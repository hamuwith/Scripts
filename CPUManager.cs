using UnityEngine;
using System.Collections.Generic;
using Unity.InferenceEngine;
using Cysharp.Threading.Tasks;

//未実装は'***'で表記

public class CPUManager : PlayerBase
{
    [SerializeField] ModelAsset modelAsset;
    Model runtimeModel;
    protected Worker worker;
    float continuousRightCount; //連続右カウント
    float continuousLeftCount; //連続左カウント
    float continuousRightRotateCount; //連続右回転カウント
    float continuousLeftRotateCount; //連続左回転カウント
    protected int[] targetPosition; //ターゲット位置
    protected int targetRotation; //ターゲット回転
    int currentRotation; //現在の回転
    Direction rotation; //回転方向
    Direction move; //移動方向
    Queue<SpecialDirection> flicks;
    const float specialMoveTime = 0.3f; //特殊移動の時間
    float specialMoveCount; //特殊移動のカウント
    enum SpecialDirection //特殊移動の方向
    {
        MoveRight,
        MoveLeft,
        RotateRight,
        RotateLeft,
    }
    public override void Start0(int id) //プレイヤーの初期化を行うメソッド
    {
        base.Start0(id);
        flicks = new Queue<SpecialDirection>();
        specialMoveCount = 0f;
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        targetPosition = new int[2];
    }
    protected override void PlayMove() //移動入力を処理するメソッド
    {
        //特殊移動待機
        if (specialMoveCount > 0)
        {
            specialMoveCount -= Time.deltaTime;
            return;
        }
        //特殊移動あり
        //while (flicks.TryDequeue(out var flick))
        //{
        //    //特殊移動の処理
        //    if (flick == SpecialDirection.RotateRight)
        //    {
        //        Rotation(Direction.Right);
        //        Rotation(Direction.Left);
        //    }
        //    else if (flick == SpecialDirection.RotateLeft)
        //    {
        //        Rotation(Direction.Left);
        //        Rotation(Direction.Right);
        //    }
        //    else if (flick == SpecialDirection.MoveRight)
        //    {
        //        Move(Direction.Right);
        //    }
        //    else if (flick == SpecialDirection.MoveLeft)
        //    {
        //        Move(Direction.Left);
        //    }
        //    specialMoveCount = specialMoveTime; //特殊移動のカウントをセット
        //}
        //ターゲット位置と現在の位置を比較して、移動方向と回転方向を決定する
        if (targetPosition[0] > currentAtomPositions[0].x)
        {
            move = Direction.Right;
        }
        else if (targetPosition[0] < currentAtomPositions[0].x)
        {
            move = Direction.Left;
        }
        else
        {
            move = Direction.None;
        }
        if (targetRotation > currentRotation)
        {
            rotation = Direction.Right;
        }
        else if (targetRotation < currentRotation)
        {
            rotation = Direction.Left;
        }
        else
        {
            rotation = Direction.None;
        }
        //回転処理
        if (rotation == Direction.Right)
        {
            //連続右回転カウントを更新
            continuousRightRotateCount += Time.deltaTime;
            //連続右カウントが一定時間を超えたら右移動
            if (continuousRightRotateCount >= M.ContinuousMoveTime)
            {
                Rotation(Direction.Right);
                currentRotation = (currentRotation + 1) % 4;
                continuousRightRotateCount = 0; //連続右カウントをリセット
            }
        }
        else
        {
            //連続右回転カウントをリセット
            continuousRightRotateCount = M.ContinuousMoveTime;
        }
        if (rotation == Direction.Left)
        {
            //連続左回転カウントを更新
            continuousLeftRotateCount += Time.deltaTime;
            //連続左回転カウントが一定時間を超えたら左移動
            if (continuousLeftRotateCount >= M.ContinuousMoveTime)
            {
                Rotation(Direction.Left);
                currentRotation = (currentRotation + 3) % 4;
                continuousLeftRotateCount = 0; //連続左カウントをリセット
            }
        }
        else
        {
            //連続左回転カウントをリセット
            continuousLeftRotateCount = M.ContinuousMoveTime;
        }
        //移動処理
        if (move == Direction.Right)
        {
            //連続右カウントを更新
            continuousRightCount += Time.deltaTime;
            //連続右カウントが一定時間を超えたら右移動
            if (continuousRightCount >= M.ContinuousMoveTime)
            {
                Move(Direction.Right);//, currentAtomPositions, stageAtom); //原子を右に移動
                continuousRightCount = 0; //連続右カウントをリセット
            }
        }
        else
        {
            //連続右カウントをリセット
            continuousRightCount = M.ContinuousMoveTime;
        }
        if (move == Direction.Left)
        {
            //連続左カウントを更新
            continuousLeftCount += Time.deltaTime;
            //連続左カウントが一定時間を超えたら左移動
            if (continuousLeftCount >= M.ContinuousMoveTime)
            {
                Move(Direction.Left);//, currentAtomPositions, stageAtom); //原子を左に移動
                continuousLeftCount = 0; //連続左カウントをリセット
            }
        }
        else
        {
            //連続左カウントをリセット
            continuousLeftCount = M.ContinuousMoveTime;
        }
        float dropValue = 0;
        if (move == Direction.None && rotation == Direction.None)
        {
            //ドロップカウントを更新
            dropValue = M.DownAcceleration * Time.deltaTime;
        }
        else
        {
            //ドロップカウントを更新
            dropValue = Time.deltaTime;
        }
        DropStep(dropValue); //ドロップ処理
    }
    protected override void SetAtoms(bool isStart = true) //原子をセットするメソッド
    {
        base.SetAtoms(isStart);
        currentRotation = 0;
        if (isStart)
        {
            //ターゲット位置を設定する
            SetTargetPosition().Forget();
        }
    }
    protected virtual async UniTaskVoid SetTargetPosition() //ターゲット位置を設定するメソッド
    {
        // Tensorに変換 追加書き込みできないため毎回新規作成
        using Tensor<float> inputField = new Tensor<float>(new TensorShape(1, 6, 16), stageAtomF);
        using Tensor<float> inputCurrentPair = new Tensor<float>(new TensorShape(1, 2));
        using Tensor<float> inputNextPair1 = new Tensor<float>(new TensorShape(1, 2));
        using Tensor<float> inputNextPair2 = new Tensor<float>(new TensorShape(1, 2));
        using Tensor<float> inputDisturberNum = new Tensor<float>(new TensorShape(1, 1));
        //Tensor にデータをセット
        for (int i = 0; i < 2; i++)
        {
            inputCurrentPair[0, i] = (int)currentAtoms[i].AtomType + 1;
            inputNextPair1[0, i] = (int)nextAtoms[i].AtomType + 1;
            inputNextPair2[0, i] = (int)nextAtoms[i + 2].AtomType + 1;
        }
        inputDisturberNum[0, 0] = gotDisturbanceNumber;
        //ワーカーにデータをセット
        worker.SetInput("field", inputField);
        worker.SetInput("current_pair", inputCurrentPair);
        worker.SetInput("next_pair1", inputNextPair1);
        worker.SetInput("next_pair2", inputNextPair2);
        worker.SetInput("disturber_num", inputDisturberNum);
        //行動の推測
        worker.Schedule();
        await UniTask.Delay(30);
        using Tensor<int> output = worker.PeekOutput() as Tensor<int>; //行動のピークを取得
        //Tensor outputs = new Tensor<int>(new TensorShape(1, 24)); //解放が必要、また使用時は型変換が必要
        //worker.CopyOutput("action_values", ref outputs); //行動の配列を取得
        //GPU使用時に必要
        //var output = output.ReadbackAndClone();
        var qValues = output.DownloadToArray();
        //確率の最大値を取得
        int action = qValues[0];
        //outputs.Dispose();
        //変換し座標と回転を取得
        targetPosition[0] = action % M.Size.x;
        targetRotation = action / M.Size.x;
        Debug.Log($"{name},x:{targetPosition[0]}, r:{targetRotation}, stage1:{dropAtomYs[1]}");
        //軸でない元素の座標を取得
        switch (targetRotation)
        {
            case 0:
                targetPosition[1] = targetPosition[0];
                break;
            case 1:
                targetPosition[1] = targetPosition[0] - 1;
                break;
            case 2:
                targetPosition[1] = targetPosition[0];
                break;
            case 3:
                targetPosition[1] = targetPosition[0] + 1;
                break;
        }
        ////特殊行動をセット
        //if (targetPosition[0] < 2 || targetPosition[1] < 2)
        //{
        //    //左側
        //    flicks.Enqueue(dropAtomYs[1] >= 11 ? SpecialDirection.RotateLeft : SpecialDirection.MoveLeft);
        //    if (targetPosition[0] < 1 || targetPosition[1] < 1)
        //    {
        //        flicks.Enqueue(dropAtomYs[0] >= 11 ? SpecialDirection.RotateLeft : SpecialDirection.MoveLeft);
        //    }
        //}
        //else if (targetPosition[0] > 2 || targetPosition[1] > 2)
        //{
        //    //右側
        //    flicks.Enqueue(dropAtomYs[3] >= 11 ? SpecialDirection.RotateRight : SpecialDirection.MoveRight);
        //    if (targetPosition[0] > 3 || targetPosition[1] > 3)
        //    {
        //        flicks.Enqueue(dropAtomYs[4] >= 11 ? SpecialDirection.RotateRight : SpecialDirection.MoveRight);
        //        if (targetPosition[0] > 4 || targetPosition[1] > 4)
        //        {
        //            flicks.Enqueue(dropAtomYs[5] >= 11 ? SpecialDirection.RotateRight : SpecialDirection.MoveRight);
        //        }
        //    }
        //}
    }
    void OnDestroy()
    {
        worker.Dispose();
    }
}

