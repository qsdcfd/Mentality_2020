## 목적: 나와 같은 사람을 위한 심리상담 대화 모델

# Korean Language Model for Wellness Conversation
`huggingface transformers`, `pytorch`, `한국어 Language Model`과 [AI 허브 정신건강 상담 데이터](http://www.aihub.or.kr/keti_data_board/language_intelligence)를 활용한 심리상담 대화 모델.  
  
## 개요 
언어모델에 대해 `auto regressive`, `text classification` 파인튜닝 및 테스트  
- **KoGPT2**: **질의**가 주어졌을 때, 다음 **답변**에 대한 텍스 생성
- **KoELECTRA**: **질의**에 대해서 **카테고리를 예측** 
- **KoBERT**:  **질의**에 대해서 **카테고리를 예측** 

## 사용 Language Model
KoELECTRA, KoBERT, KoGPT2

## 환경
### Data
- [AI 허브 정신건강 상담 데이터](http://www.aihub.or.kr/keti_data_board/language_intelligence): 심리 상담 데이터의 경우 회원가입 후 신청하면 다운로드 가능.
- [songys/Chatbot_data](https://github.com/songys/Chatbot_data)
### GPU
Colab pro, P100
### Package
```
kogpt2-transformers
kobert-transformers
transformers==3.0.2
torch
```

## Task
### 1. KoELECTAR & KoBERT Text Classifcation
KoELECTAR 및 KoBERT를 이용한 텍스트 분류 모델.
#### 1.1 질의에 대한 카테고리 분류
##### 데이터
Wellness 심리 상담 데이터 사용. Wellness 데이터의 경우 **카테고리/ 질문/ 답변**으로 나누어져있다. 카테고리 별로 3개 내외의 답변을 가지고 있으므로
Wellness 데이터의 경우  질문과 카테고리 클래스의 쌍으로 만들어 학습.   
  
**카테고리 클래스** 데이터: 카테고리 클래스 `359`개
```txt
감정/감정조절이상    0
감정/감정조절이상/화    1
감정/걱정    2
```
**카테고리 클래스와 질의** 데이터  
```text
감정/감정조절이상    그럴 때는 밥은 잘 먹었는지, 잠은 잘 잤는지 체크해보는 것도 좋아요.
감정/감정조절이상/화    화가 폭발할 것 같을 때는 그 자리를 피하는 것도 좋은 방법이라고 생각해요.
감정/걱정    모든 문제는 해결되기 마련이잖아요. 마음을 편히 드세요.
```
  
**질의과 카테고리 클래스 쌍** 데이터: 5231
```txt
순간순간 감정조절을 못해요.    0
예전보다 화내는 게 과격해진 거 같아.    1
수술한다는 말에 얼마나 걱정이 되던지…    2
```

<br>

##### 모델

###### 1.KoELECTRA
```python
class koElectraForSequenceClassification(ElectraPreTrainedModel):
  def __init__(self,
               config,
               num_labels):
    super().__init__(config)
    self.num_labels = num_labels
    self.electra = ElectraModel(config)
    self.classifier = ElectraClassificationHead(config, num_labels)

    self.init_weights()
...생략...
```

<br>

###### 2.KoBERT
```python
class KoBERTforSequenceClassfication(BertPreTrainedModel):
  def __init__(self,
                num_labels = 359, # 분류할 라벨 갯수를 설정
                hidden_size = 768, # hidden_size
                hidden_dropout_prob = 0.1,  # dropout_prop
               ):
    super().__init__(get_kobert_config())

    self.num_labels = num_labels 
    self.kobert = get_kobert_model()
    self.dropout = nn.Dropout(hidden_dropout_prob)
    self.classifier = nn.Linear(hidden_size, num_labels)

    self.init_weights()
...생략...
```

<br>

### 2. KoGPT2 Text Generation(Auto Regressive)
GPT-2 모델을 이용한 대화 및 답변 텍스트 생성.

#### Text Generation

##### 데이터
**카테고리/ 질문/ 답변** 데이터에서 질문과 답변의 쌍으로 구성.

```text
꼭 롤러코스터 타는 것 같아요.    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
꼭 롤러코스터 타는 것 같아요.    저도 그 기분 이해해요. 많이 힘드시죠?
롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
롤러코스터 타는 것처럼 기분이 왔다 갔다 해요.    저도 그 기분 이해해요. 많이 힘드시죠?
작년 가을부터 감정조절이 잘 안 되는 거 같아.    감정이 조절이 안 될 때만큼 힘들 때는 없는 거 같아요.
작년 가을부터 감정조절이 잘 안 되는 거 같아.    저도 그 기분 이해해요. 많이 힘드시죠?
```

##### 모델
```python
class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = get_kogpt2_model()
    
  
...생략...
```

##### 텍스트 생성부분


[how-to-generate-text](https://huggingface.co/blog/how-to-generate?fbclid=IwAR2BZ4BNG0PbOvS5QaPLE0L3lx7_GOy_ePVu4X1LyTktQo-nLEPr7eht1O0) 참고 하여, Huggingface의 Generate 사용. 
```python
def generate(self,
               input_ids,
               do_sample=True,
               max_length=50,
               top_k=0,
               temperature=0.7):
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_k=top_k,
               temperature=temperature)
```

<br>

### 3. 결과

**KoELECTRA** 
```
Question: 벽에 머리를 부딪히는 느낌이야
Answer: 머리가 아프면 정말 괴롭죠. 진통제를 한 알 정도 먹어 보는 건 어떨까요?, index: 334, value: 7.020100116729736
--------------------------------------------------

Question: 허리가 아파서 움직임이 어렵네ㅠㅠ
Answer: 정말 힘드셨겠어요. 의사의 도움을 받아 보는 건 어떨까요?, index: 333, value: 9.480134010314941
--------------------------------------------------

Question: 집중력도 떨어지고 기분이 좋지 않아
Answer: 집중이 잘 안 될 때가 있죠. 그럴 때는 아예 잠깐 다른 일을 하는 것도 좋아요., index: 325, value: 6.232089996337891
--------------------------------------------------

Question: 나는 화가 통제가 안돼!
Answer: 저도 그 기분 이해해요. 많이 힘드시죠?, index: 0, value: 6.6863813400268555

```

<br>

**KoGPT2**  
KoGPT2의 경우 Text Generation 방법에 따라 퀄리티가 좌우된다. 각각에 맞는 적절한 설정 변경 필요.
```txt
Question: 다이어트 하고싶은데 맘처럼 안되네
Answer: 다이어트 파이팅! 시작해봐요. 완벽한100%를 만드는 건 쉽지 않은 일이죠. 그 마음 다 잡았다면 후회없게 잊을 거예요. 포이즌을 해보세요.
----------------------------------------------------------------------------------------------------
Question: 요즘은 이상한 생각이 많이 들어
Answer: 그럴 때가 있죠. 부정적인 생각이 또 다른 부정적인 생각을 불러올 때가 있어요. 그 건 나쁜 거라고 생각해요. 나쁜 생각을 많이 하는 건 어떨까요? 먼저 부정적인 생각을 해보는 건 어떨까요? 다시 좋아질
----------------------------------------------------------------------------------------------------
Question: 부정적인 생각이 많이 드네
Answer: 부정적인 생각을 하는 횟수가 많아졌나봅니다.긍정적인 생각을 해보세요.긍정적인 생각을 해보세요. 그러면 조금 더 빨리 낫고 더 행복해질 거예요.항상 기뻐하시는 모습을 보니 저도 기뻐요.
----------------------------------------------------------------------------------------------------
Question: 사고 휴유증이 있는걸까
Answer: 그런가요?그런 일이 있으셨군요. 확신을 못하시는군요. 확신을 가지신다면 조금은 거리를 두는 것도 좋을 것 같아요. 그럴 수 있지요. 어느 정도 거리를 두셨나요?
----------------------------------------------------------------------------------------------------
Question: 체력이 떨어져서 문제야
Answer: 운동을 해보세요. 하듯이 자연스럽게 대화를 이어나가는 게 좋답니다.재미있어 보여요. 이야기를 이어나가면서 끝을 맺는 것도 좋겠죠. 이야기를 이어나가면서 끝을 맺는 것도 좋겠죠.그렇게 하면
```

<br>


**KoBERT**
```
Question: 입맛도 통 없구
Answer: 그래도 끼니는 거르시면 안 돼요. 작은 거라도 조금씩 드세요., index: 309, value: 6.560961723327637
--------------------------------------------------

Question: 기분이 우울해서 큰일이야
Answer: 우울할 때는 칭찬타임! 오늘의 잘한 일을 말해봐요., index: 59, value: 8.06270694732666
--------------------------------------------------
```

<br>

## 기간
2020.08~2021.02
# References
[KoBERT](https://github.com/SKTBrain/KoBERT)  
[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)  
[KoGPT2](https://github.com/SKT-AI/KoGPT2)  
[KoGPT2-Transformers](https://github.com/taeminlee/KoGPT2-Transformers/)  
[KoELECTRA](https://github.com/monologg/KoELECTRA)  
[enlipleai/kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)  
[how-to-generate-text](https://huggingface.co/blog/how-to-generate?fbclid=IwAR2BZ4BNG0PbOvS5QaPLE0L3lx7_GOy_ePVu4X1LyTktQo-nLEPr7eht1O0)
