# 맹검 EEG 창 판독 지침

평가자는 임상결과, SR 결과, 계산된 알파 파워 및 실제 VitalDB case ID를
알 수 없어야 합니다. 각 이미지는 무작위 `review_id`로만 식별됩니다.

## 판독 질문

1. 표시된 120초 구간에 명백한 artifact가 없는가?
2. 표시된 120초 구간에 burst suppression 또는 suppression-like activity가 없는가?
3. 표시된 120초 구간이 유도, 각성 또는 suppression으로 이행하는 뚜렷한 전환 구간이 아닌가?
4. 두 EEG 채널이 모두 판독 가능한가?
5. 위 조건을 모두 충족하면 `ACCEPT`, 그렇지 않으면 `REJECT`로 기록한다.

알파 리듬의 크기, 뚜렷함, 기울기 또는 존재 여부는 판정에 사용하지
않습니다. 알파가 작거나 뚜렷하지 않다는 이유만으로 창을 거부해서는
안 됩니다.

## 판독 파일

`reviewer_form.csv`의 다음 열을 작성합니다.

- `decision`: `ACCEPT`, `REJECT`, `UNCERTAIN`
- `artifact_present`: `0` 또는 `1`
- `suppression_present`: `0` 또는 `1`
- `transition_present`: `0` 또는 `1`
- `both_channels_interpretable`: `0` 또는 `1`
- `comment`: 필요한 경우에만 간단히 기록

주분석에 포함될 모든 창은 최소 한 명의 맹검 평가자가 판독합니다.
두 번째 평가자는 무작위 표본을 독립적으로 판독하여 평가자 간 일치도를
검증합니다.

