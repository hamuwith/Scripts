using DG.Tweening;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Threading;
using Cysharp.Threading.Tasks;

public class TestDotween : MonoBehaviour
{
    [SerializeField] MonoBehaviour TestMonoBehaviour;
    public List<GameObject> TweenTargetObject;
    List<StartData> startDatas;
    public Sequence DOTweenTest()
    {
        IDOTweenTest TweenTarget = TestMonoBehaviour as IDOTweenTest;
        startDatas = new List<StartData>();
        foreach (var obj in TweenTargetObject)
        {
            StartData startData = new StartData();
            startData.SetStartData(obj);
            startDatas.Add(startData);
        }
        var seq = TweenTarget.DOTweenTest();
        _ = End(seq);
        return seq;
    }
    async UniTask End(Sequence sequence)
    {
        await sequence.AsyncWaitForCompletion();
        
        foreach (var startData in startDatas)
        {
            startData.GetStartData();
        }
    }
}
public interface IDOTweenTest
{
    Sequence DOTweenTest();
}
//オブジェクトの初期値管理
[System.Serializable]
public class StartData
{
    GameObject gameObject;
    Vector3 position;
    Vector3 scale;
    Vector3 rotate;
    Color color;
    string text;
    Sprite sprite;
    TextMeshProUGUI textMeshProUGUI;
    TextMeshPro textMeshPro;
    Image image;
    SpriteRenderer spriteRenderer;
    bool enabled;
    bool isActive;
    public void SetStartData(GameObject gameObject)
    {
        this.gameObject = gameObject;
        textMeshProUGUI = gameObject.GetComponent<TextMeshProUGUI>();
        textMeshPro = gameObject.GetComponent<TextMeshPro>();
        image = gameObject.GetComponent<Image>();
        spriteRenderer = gameObject.GetComponent<SpriteRenderer>();
        //初期値を取得
        position = gameObject.transform.position;
        scale = gameObject.transform.localScale;
        rotate = gameObject.transform.localEulerAngles;
        isActive = gameObject.activeSelf;
        //ターゲットオブジェクトがnullでない場合、初期値を取得
        if (textMeshProUGUI != null)
        {
            text = textMeshProUGUI.text;
            color = textMeshProUGUI.color;
            enabled = textMeshProUGUI.enabled;
        }
        else if (textMeshPro != null)
        {
            text = textMeshPro.text;
            color = textMeshPro.color;
            enabled = textMeshPro.enabled;
        }
        else if (image != null)
        {
            sprite = image.sprite;
            color = image.color;
            enabled = image.enabled;
        }
        else if (spriteRenderer != null)
        {
            sprite = spriteRenderer.sprite;
            color = spriteRenderer.color;
            enabled = spriteRenderer.enabled;
        }
    }
    public void GetStartData()
    {
        if (!gameObject.activeSelf) gameObject.SetActive(true);
        //初期値を設定
        gameObject.transform.position = position;
        gameObject.transform.localScale = scale;
        gameObject.transform.localEulerAngles = rotate;
        //ターゲットオブジェクトがnullでない場合、初期値を設定
        if (textMeshProUGUI != null)
        {
            textMeshProUGUI.text = text;
            textMeshProUGUI.color = color;
            textMeshProUGUI.enabled = enabled;
        }
        else if (textMeshPro != null)
        {
            textMeshPro.text = text;
            textMeshPro.color = color;
            textMeshPro.enabled = enabled;
        }
        else if (image != null)
        {
            image.sprite = sprite;
            image.color = color;
            image.enabled = enabled;
        }
        else if (spriteRenderer != null)
        {
            spriteRenderer.sprite = sprite;
            spriteRenderer.color = color;
            spriteRenderer.enabled = enabled;
        }
        if (gameObject.activeSelf != isActive) gameObject.SetActive(isActive);
    }
}