# 연구 단계 종료 결정 — 원본 E2O1 게이트 완료 아님

연구 단계는 종료한다. 원본 E2O1 게이트는 완료가 아니다.
원본 #142 체크리스트 14개 DoD는 미충족 상태로 보존된다.

## 표준 DER (공식 md-eval-22, BUT only_words, strict 0, UEM 68.22s, n=4 train)

| arm | DER | miss | FA | conf |
|---|---|---|---|---|
| PSEM | 24.05% | 20.18% | 2.36% | 1.50% |
| Soniox MAN LEXICAL | 60.47% | 56.73% | 2.97% | 0.77% |
| Soniox MAN TURN (진단) | 22.93% | 18.41% | 2.97% | 1.55% |

PSEM−LEXICAL = −36.42pp, PSEM−TURN = +1.12pp. ±250ms: PSEM 21.41%,
TURN 17.51%. Soniox lexical MISS(56.73%)는 출력 간극이며 모델 승리가 아니다.
TURN 간극 메우기가 MISS를 지운 출력 변환 차이이다.

## NP2 PRIMARY (동일 7s, 음성 4.65s, 타깃 [33405760,33517760))

| arm | DER | miss | FA | conf |
|---|---|---|---|---|
| CONT warm | 25.59% | 1.00 | 0.19 | 0.00 |
| PHASE_CHUNK cold | 61.72% | 0.80 | 0.45 | 1.62 |

Delta −36.13pp. qc=33400320(340ms 실측 prior) vs 전구간 0~35분 상태,
둘 다 80ms+480ms 글로벌 청크 정렬. ORIG 58.49와 PHASE frame-only 43.44는
역사 기록이며 인과 primary가 아니다. 효과는 history+state joint이며,
warm conf 0이나 MISS 악화로 일반 수정이 아니다.

## 고정 사항

- Soniox는 TARGET 성능 상한이며 수학적 상한이 아니다.
- PSEM은 오픈소스 ASR-agnostic이다. 현 후보 Sortformer 4spk v2.1
  GGUF 62faec7b / exe 3706은 테스트 후보이며 새로 학습한 헤드가 아니다.
- 기존 198% 비율은 일반 성능에 대해 철회한다.
- 물리 ASR 턴 / 논리 소유권 턴 / 모델 캐시는 수명이 다르다.
  격리 state0만으로 prod ASR 턴 리셋을 증명하지 못한다.
- 연속 프로듀서는 실측이나 decode_events/제품 통합은 미테스트이다.
- 단어 시작 정정 39→29는 측정 타임스탬프 수정이며 번역 이익이 아니다.
- H7301 가중치 손실 / DEV 꼬리 복사 정정은 보존, 새 학습 주장 없음.
- 12 에피소드를 전 항목 완료로 표기하지 않는다.

## 출처
- scorer md-eval-22 sha 872aa955, refs BUT 2509d893. 상세 수치·입력·재현은
  `EVIDENCE.json` + `replay.py`(54회 공식 실행) 참조. 과거 freeze 오류는
  철회 기록으로 보존되며 현행으로 주장하지 않는다.

## 재현·커밋 범위

본 커밋은 연구 정리이며 코드 범위는 재현·검증용 compact 번들에 한한다.
GitHub issue 상태 변경 없음.

점수 54회 재현: `python experiments/psem_research_closeout/replay.py`
아카이브 검증 포함: `python experiments/psem_research_closeout/replay.py --verify-archive`
요구사항은 Perl과 표준 라이브러리뿐이다. 점수 재현에는 비공개 아카이브·GPU가
필요 없다. 전체 native/ASR 재생성은 비공개 아카이브·원본·모델이 별도로 필요하다.
푸시 대상은 compact 12파일이며 원본 ZIP은 푸시하지 않는다.

## 범위·행정 상태

- 이 정리는 로컬 아티팩트와 재현 코드에 한정하며, GitHub 이슈 상태를 추가로 변경하지 않는다.
- #142 CLOSED(진단 공지), #155 OPEN(기생성). 추가 이슈 생성·종료·편집 보류.
  본 정리는 #155 확장이 아니며, 인과 소유권 잔여분은 미검증 상태로 남는다.
- 6M 기존 수정분·유료 캡처·외부 corpus/model·타 worktree를 건드리지 않는다.
