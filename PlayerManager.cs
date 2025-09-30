using UnityEngine;
using UnityEngine.InputSystem;
using System.Linq;
//未実装は'***'で表記

public class PlayerManager : PlayerBase
{
    [SerializeField] InputActionAsset inputActions; //入力アクションアセット
    [SerializeField] float moveSensitivity; //入力の感度
    Vector2 moveInput; //移動入力
    float rotateInput; //回転入力
    float continuousRightCount; //連続右カウント
    float continuousLeftCount; //連続左カウント
    float continuousRightRotateCount; //連続右回転カウント
    float continuousLeftRotateCount; //連続左回転カウント
    public override void Start0(int id) //プレイヤーの初期化を行うメソッド
    {
        base.Start0(id);
        //操作の初期化
        SetInputEvent();
    }
    private void SetInputEvent() //入力イベントを設定するメソッド
    {
        var actionMaps = inputActions.actionMaps.ToDictionary(x => x.name, x => x);
        actionMaps.TryGetValue("Player", out var playerMap);
        var actionPlayer = playerMap.ToDictionary(x => x.name, x => x);
        actionPlayer.TryGetValue("Move", out var move);
        actionPlayer.TryGetValue("Drop", out var drop);
        actionPlayer.TryGetValue("Rotation", out var rotation);
        actionPlayer.TryGetValue("Pause", out var pause);
        move.performed += context => moveInput = context.ReadValue<Vector2>();
        move.canceled += context => moveInput = context.ReadValue<Vector2>();
        drop.started += context => Drop();
        rotation.performed += context => rotateInput = context.ReadValue<float>();
        rotation.canceled += context => rotateInput = context.ReadValue<float>();
        pause.performed += context => M.Pause();
    }
    protected override void PlayMove() //移動入力を処理するメソッド
    {
        //回転処理
        if (rotateInput > 0.5f)
        {
            //連続右回転カウントを更新
            continuousRightRotateCount += Time.deltaTime;
            //連続右カウントが一定時間を超えたら右移動
            if (continuousRightRotateCount >= M.ContinuousMoveTime)
            {
                Rotation(Direction.Right);
                continuousRightRotateCount = 0; //連続右カウントをリセット
            }
        }
        else
        {
            //連続右回転カウントをリセット
            continuousRightRotateCount = M.ContinuousMoveTime;
        }
        if (rotateInput < -0.5f)
        {
            //連続左回転カウントを更新
            continuousLeftRotateCount += Time.deltaTime;
            //連続左回転カウントが一定時間を超えたら左移動
            if (continuousLeftRotateCount >= M.ContinuousMoveTime)
            {
                Rotation(Direction.Left);
                continuousLeftRotateCount = 0; //連続左カウントをリセット
            }
        }
        else
        {
            //連続左回転カウントをリセット
            continuousLeftRotateCount = M.ContinuousMoveTime;
        }
        //移動処理
        if (moveInput.x > moveSensitivity)
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
        if (moveInput.x < -moveSensitivity)
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
        if (moveInput.y < -moveSensitivity)
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
}

