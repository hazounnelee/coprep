# Data Pipeline Refactor Design

## Background

기존 전처리 파이프라인이 라인별 폴더 + LOT별 엑셀 파일 구조를 전제로 설계되어 있으나, 데이터 제공 형태가 통합 파일 기반으로 변경됨. 또한 원재료 가중합(`STEP_MATERIAL_`) 피처 생성 로직이 누락된 상태.

## Goals

1. 새 데이터 폴더/파일 구조에 맞게 로딩·트래킹·전처리 수정
2. 누락된 원재료 가중합 로직 구현
3. `STEP_METERIAL_` → `STEP_MATERIAL_` 오타 수정
4. `--debug` 플래그 추가 (컬럼 매칭 실패 진단)

## Non-Goals

- 파이프라인 구조 전면 리팩터링 (loader → tracker → preprocessor 유지)
- XGBoost 학습 파이프라인 변경
- 새로운 피처 추가 (LOT_STEP의 입도/밀도/성분 컬럼은 공란이므로 사용 안 함)

## Data Structure Changes

### Old Structure

```
데이터폴더/
  {N}라인/                          ← 1라인~10라인
    통합일지/*.xlsx                  ← 1시트
    수기운전일지/*.xlsx              ← 1시트
    제품시험검사이력/*.xlsx
    제품검사판정이력/*.xlsx
    용해작업실적/*.xlsx              ← 2시트, Sheet1(▶마커) + Sheet2(원료)
    반응투입스케쥴실적/*.xlsx        ← 3시트, Sheet1(▶마커) + Sheet2(투입) + Sheet3(초기조건)
  원재료/
    가성소다/*.xlsx
    황산코발트/*.xlsx
    황산망간/*.xlsx
    황산니켈/*.xlsx
    암모니아/*.xlsx
  검사이력/
    검사이력/*.xlsx
    제품검사판정이력/*.xlsx
    제품시험검사이력/*.xlsx
```

### New Structure

```
데이터폴더/
  통합일지/
    복사본 2024년도 1라인-전구체2단계 통합일지.xlsx
    복사본 2024년도 2라인-전구체2단계 통합일지.xlsx
    ...
  수기운전일지/
    1라인/*.xlsx                    ← 파일명 무의미, LOT으로 매칭
    2라인/*.xlsx
    ...
  용해작업실적/
    MELT_WRK_ORD_METAL.xlsx         ← 모든 라인 통합
    MELT_WRK_ORD.xlsx               ← 모든 라인 통합
  반응투입스케줄/
    반응투입_LOT_INIT.xlsx          ← 모든 라인 통합
    반응투입_LOT_STEP.xlsx          ← 모든 라인 통합
  원재료/
    MATR_{TYPE}_{5자리코드}_원료검사판정이력_220401_260408.xlsx
    (TYPE: COSO4, MNSO4, NISO4, NAOH, NH4OH)
```

### Removed Data

- `제품시험검사이력`, `제품검사판정이력`, `검사이력`: 로드만 하고 사용처 없었으므로 제거
- `TASKS_TESTRESULT`, `TASKS_1SHEET` 등 관련 상수 정리

## File-by-File Changes

### loader.py

기존 `_load_1sheet`, `_load_2sheet`, `_load_3sheet`, `get_alldata` 전면 교체.

반환 구조 변경:
- 기존: `defaultdict[라인][태스크] → list[DataFrame or dict]`
- 신규: `dict[데이터종류] → DataFrame or dict`

```python
def get_alldata(path_folder: str, debug: bool = False) -> dict:
    return {
        "통합일지": _load_integrated(path_folder, debug),
        # → {"1라인": df, "2라인": df, ...}

        "수기운전일지": _load_handrecorded(path_folder, debug),
        # → {"1라인": [df, df, ...], ...}

        "용해_metal": _load_melt_metal(path_folder, debug),
        # → DataFrame (MELT_WRK_ORD_METAL, header=1)

        "용해_ord": _load_melt_ord(path_folder, debug),
        # → DataFrame (MELT_WRK_ORD, header=1)

        "반응_init": _load_react_init(path_folder, debug),
        # → DataFrame (반응투입_LOT_INIT)

        "반응_step": _load_react_step(path_folder, debug),
        # → DataFrame (반응투입_LOT_STEP, header=[0,1])

        "원재료": _load_materials(path_folder, debug),
        # → {"NAOH": df, "COSO4": df, "MNSO4": df, "NISO4": df, "NH4OH": df}
        # 각 df는 입고LOT + 필요 물성 컬럼만 포함
    }
```

통합일지: 파일명에서 라인 번호 추출 (정규식으로 `(\d+)라인` 매칭).

수기운전일지: 라인 하위폴더 순회, 각 파일 로드.

용해작업실적: 두 파일 모두 1행째 `1.화면항목` 헤더 스킵 (`header=1`).

반응투입 LOT_STEP: 2행 헤더 (`header=[0,1]`). 이후 Unnamed 컬럼 매핑.

원재료: 3행 헤더. 파일명 prefix에서 종류 구분 (`MATR_NAOH_` 등). 필요 물성 컬럼만 추출:
- `입고LOT` (매칭 키)
- CoSO4: `Chemical Composition(C01) > Co`
- MnSO4: `Chemical Composition(C01) > Mn`, `Initial PH(C03) > pH`
- NiSO4: `Chemical Composition(C01) > Ni`
- NaOH: `Chemical Composition(C01) > NaOH`

대소문자 무시로 컬럼 매칭.

debug 모드: 기대 컬럼이 실제 파일에 없을 때 `[DEBUG]` 경고 출력.

### tracker.py

구조 변경: `dict_lines_tracked` (라인별 dict) → `df_tracked` (단일 DataFrame).

```python
class TrackerRawData:
    def __init__(self, data: dict, list_lines, product_name, debug=False):
        df_lots = self._extract_all_lot_pairs(data["통합일지"], list_lines, product_name)
        df = self._attach_reacted(df_lots, data["반응_init"], data["반응_step"])
        df = self._attach_handrecorded(df, data["수기운전일지"])
        df = self._attach_melting(df, data["용해_metal"], data["용해_ord"])
        df = self._attach_materials(df, data["원재료"])
        self.df_tracked = df
```

#### _extract_all_lot_pairs

각 라인의 통합일지에서 `(lot_reacted, lot_target)` 쌍 추출 후 전체 합침. 기존 `_extract_lot_pairs` 로직 유지.

#### _attach_reacted

- 기존: lot_reacted에서 날짜+설비 파싱 → 반응투입스케쥴실적 파일 순회 매칭
- 신규: `반응_init`/`반응_step`에서 `생산LOT번호`로 직접 필터
- `반응_step`에서 추출: weights_metal/naoh/nh4oh, lots_metal/naoh/nh4oh, steps_ph/rpm, steps_num/time
- `반응_init`에서 추출: 초기조건 값 (preprocessor에서 직접 사용)

컬럼 매핑 (반응_step):
| 기존 (Sheet2) | 신규 (LOT_STEP) |
|---|---|
| Unnamed: 6 (Metal 투입량) | Unnamed: 5 |
| Unnamed: 7 (Metal LOT) | Unnamed: 6 |
| Unnamed: 10 (NaOH 투입량) | Unnamed: 8 |
| Unnamed: 11 (NaOH LOT) | Unnamed: 9 |
| Unnamed: 14 (NH4OH 투입량) | Unnamed: 11 |
| Unnamed: 15/19 (NH4OH LOT) | Unnamed: 12 |
| PH | PH |
| 교반기\nRPM | 교반기RPM |

1라인 NH4OH LOT 예외(Unnamed:19)는 제거 — 신규 데이터에서는 모든 라인이 Unnamed:12.

#### _attach_handrecorded

기존과 동일. 수기운전일지 라인 폴더에서 LOT 매칭.

#### _attach_melting

- 기존: 라인별 용해작업실적 dict에서 `lot_melted` 매칭, Sheet2에서 원료명/LOT NO/투입중량
- 신규: `용해_metal`에서 `반응생산 LOT 번호`로 해당 rows 필터 → `원료명`으로 황산코발트/망간/니켈 구분, `LOT NO`와 `투입중량` 추출

#### _attach_materials

기존과 동일한 흐름. 용해에서 얻은 원재료 LOT → `data["원재료"]`에서 `입고LOT`으로 매칭. 단, 행 전체 대신 필요 물성만 가져옴.

### preprocessor.py

#### 초기조건 (INIT_COLUMNS)

10개 → 9개. IMPELLAR교반속도 제거.

컬럼명 매핑 변경 (SHEET3_COLS → INIT_COL_MAP):
| 피처명 | 기존 Sheet3 컬럼 | 신규 LOT_INIT 컬럼 |
|---|---|---|
| 반응투입_초기조건_작업시간 | 작업\n시간 | 작업시간 |
| 반응투입_초기조건_순수온도 | 순수\n온도 | 순수온도 |
| 반응투입_초기조건_순수투입중량(kg) | 순수\n투입 중량(kg) | 순수투입중량(kg) |
| 반응투입_초기조건_NAOH투입중량(kg) | NAOH\n투입중량(kg) | NAOH순수투입중량(kg) |
| 반응투입_초기조건_NH4OH투입중량(kg) | NH4OH\n투입중량(kg) | NH4OH순수투입중량(kg) |
| 반응투입_초기조건_용존산소량 | 용존산소량 | 용존산소량 |
| 반응투입_초기조건_pH | pH | Ph |
| 반응투입_초기조건_N2주입유량 | N2\n주입 유량 | N2주입유량 |
| 반응투입_초기조건_N2PURGE시간 | N2\nPURGE 시간 | N2 PURGE시간 |

대소문자 무시로 매칭.

#### 초기조건 데이터 소스 변경

기존: `row["df_reacted"]["Sheet3"]`에서 읽음 (tracker가 반응투입 전체 시트를 저장).
신규: tracker가 `반응_init` DataFrame 참조를 저장하거나, 생산LOT번호로 `반응_init`에서 직접 조회.

#### 가중합 (STEP_MATERIAL_) — 신규 구현

각 배치의 스텝 $j$마다, 그 시점까지 누적 투입된 원재료 LOT들의 물성을 가중평균:

$$STEP\_MATERIAL\_\{step\}\_\{mat\}\_\{prop\} = \frac{\sum_k w_k \cdot p_k}{\sum_k w_k}$$

tracker의 `step_info`에 저장된 LOT별 누적 투입량($w_k$)과, `data["원재료"]`에서 조회한 물성($p_k$)을 사용.

대상 물성 5개 (기존 `MATERIAL_FILTER_TERMS` 그대로 유지, 변경하지 않음):
| 피처 접미사 | 원재료 종류 | 물성 컬럼 | 비고 |
|---|---|---|---|
| 황산망간_성분_ChemicalComposition(C01)_Mn | MNSO4 | Mn | 가중합 |
| 황산망간_성분_InitialPH(C03)_pH | MNSO4 | pH | 가중합 |
| 황산니켈_성분_ChemicalComposition(C01)_Ni | NISO4 | Ni | 가중합 |
| 가성소다_성분_ChemicalComposition(C01)_NaOH | NAOH | NaOH | 가중합 |
| 황산코발트_투입량 | COSO4 | (투입중량 자체) | 기존 그대로 유지 |

NH4OH 원재료: 기존에도 `MATERIAL_FILTER_TERMS`에 없으므로 가중합 대상 아님. 로드만 하고 미사용 (기존 동작 유지).

결과 컬럼: `STEP_MATERIAL_{01~60}_{물성접미사}` (5물성 × 60스텝 = 300컬럼)

#### 수기운전일지 보간

기존과 동일. 변경 없음.

### postprocessor.py

`STEP_METERIAL_` → `STEP_MATERIAL_` 리네이밍:
- `LIST_PREFIX`: `"STEP_METERIAL_"` → `"STEP_MATERIAL_"`
- `PATTERN_METERIAL` → `PATTERN_MATERIAL`: 정규식도 업데이트
- `MATERIAL_FILTER_TERMS`: 기존 유지 (물성 접미사 기준 필터)
- `filter_material`: `like="STEP_METERIAL"` → `like="STEP_MATERIAL"`
- `postprocess_by_product`: `"STEP_METERIAL_"` → `"STEP_MATERIAL_"`

### preprocess.py

- `--debug` 플래그 추가
- `get_alldata` 호출에 `debug` 전달
- `TrackerRawData` API 변경에 맞춤 (반환값이 `df_tracked` 단일 DataFrame)
- `DataPreprocesser` API 변경에 맞춤 (라인별 dict 대신 단일 DataFrame 입력)
- 출력 형식은 기존과 동일 (`data_preprocessed.pkl`, `data_raw.pkl`, `config.txt`)

### config/schema.py

`ProductConfig.lines` 유지 — 통합일지(파일명에서 라인 구분), 수기운전일지(라인 하위폴더)에서 사용.

### tests/

기존 테스트를 새 데이터 구조에 맞게 업데이트. 테스트 픽스처의 데이터 형태 변경.

## Debug Mode

`--debug` 플래그 활성 시 `[DEBUG]` 접두사로 진단 정보 출력:

```
[DEBUG] loader: 원재료 MATR_NAOH_500LC: 기대 컬럼 'xxx' 없음. 실제 컬럼: [...]
[DEBUG] loader: 반응투입_LOT_INIT: 기대 컬럼 'IMPELLAR교반속도' 없음 (무시)
[DEBUG] tracker: LOT 'N86L-1A250504-1' 반응투입 매칭 실패
[DEBUG] tracker: LOT 'N86L-1A250504-1' 수기운전일지 없음
[DEBUG] preprocessor: 초기조건 컬럼 'Ph' → '반응투입_초기조건_pH' 매핑 완료
```

debug=False (기본)이면 기존처럼 `error_log`에만 기록.

## Output

전처리 결과물 형태 변경:

| 컬럼 그룹 | 개수 | 변경 |
|---|---|---|
| lot_target, lot_reacted | 2 | 유지 |
| 초기조건 | 9 | 10 → 9 (IMPELLAR 제거) |
| STEP_WEIGHT/PH/RPM | 300 | 유지 |
| STEP_SIZE (입도 보간) | 300 | 유지 |
| **STEP_MATERIAL (가중합)** | **300** | **신규** |

총 약 911개 컬럼.
