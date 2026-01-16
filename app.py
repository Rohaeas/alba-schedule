import streamlit as st
import pandas as pd
import random
import json
import hashlib

# ============================================================
# 0) 앱 기본 설정
# ============================================================
st.set_page_config(page_title="알바 스케줄러", layout="wide")
st.title("알바 스케줄러")

st.markdown("""
<style>
/* 전체 여백 (상단 여백 복구) */
.block-container {
    padding-top: 2.2rem;   /* ← 핵심 수정 */
    padding-bottom: 2.2rem;
}

/* 제목 자간 */
h1, h2, h3 {
    letter-spacing: -0.3px;
}

/* 버튼 */
.stButton > button {
    border-radius: 12px;
    padding: 0.55rem 0.95rem;
    font-weight: 600;
}

/* 입력창 */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}

/* Expander 카드 */
div[data-testid="stExpander"] {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.02);
}

/* DataFrame 카드 */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

요일들 = ["월", "화", "수", "목", "금", "토", "일"]
평일 = ["월", "화", "수", "목", "금"]
주말 = ["토", "일"]


# ============================================================
# 1) 세션 상태(Session State) 초기화
#    - Streamlit은 위젯 조작마다 스크립트를 재실행(rerun)함.
#    - 따라서 "현재 설정 값"은 st.session_state에 보관해야 안정적.
# ============================================================
def 세션초기화():
    st.session_state.setdefault("운영방식", "3조(오픈/미들/마감)")

    # 알바 명단(확정 저장본)
    st.session_state.setdefault("알바수", 3)
    st.session_state.setdefault("알바생저장", [])

    # 알바 입력 폼 위젯 key 갱신용(불러오기/적용 후 “한 번에 반영 안됨” 방지)
    st.session_state.setdefault("알바폼버전", 0)

    # 근무 옵션
    st.session_state.setdefault("옵션_같은날허용", False)
    st.session_state.setdefault("옵션_하루최대", 1)
    st.session_state.setdefault("옵션_인접만", True)

    # 필요인원(표/딕트)
    st.session_state.setdefault("필요인원표", None)  # DataFrame
    st.session_state.setdefault("필요인원", None)    # dict[요일][조] -> int

    # JSON 업로드 무한 rerun 방지용(같은 파일은 1회만 적용)
    st.session_state.setdefault("마지막_json_해시", None)

세션초기화()


# ============================================================
# 2) 운영방식 / 데이터 구조 유틸
# ============================================================
def 조리스트_가져오기(운영방식: str):
    """운영방식(3조/2조/1인)에 따라 조 리스트를 반환."""
    if 운영방식 == "3조(오픈/미들/마감)":
        return ["오픈", "미들", "마감"]
    if 운영방식 == "2조(오픈/마감)":
        return ["오픈", "마감"]
    return ["근무"]


def 기본_필요인원표(조들):
    """기본 필요인원표(모든 셀 1) 생성."""
    return pd.DataFrame(1, index=조들, columns=요일들, dtype="int64")


def 필요인원표_동기화(조들):
    """
    운영방식이 바뀌어서 조들이 달라져도,
    기존 표의 값(가능한 범위)은 유지하고 새 표로 안전하게 변환.
    """
    기존 = st.session_state["필요인원표"]
    if 기존 is None:
        st.session_state["필요인원표"] = 기본_필요인원표(조들)
        return

    새 = pd.DataFrame(0, index=조들, columns=요일들, dtype="int64")

    # 공통으로 존재하는 조는 기존 값을 이어받음
    for 조 in set(조들).intersection(set(기존.index)):
        for 요일 in 요일들:
            새.loc[조, 요일] = int(기존.loc[조, 요일])

    # 새로 생긴 조는 기본 1로 채움
    for 조 in 조들:
        if 조 not in 기존.index:
            새.loc[조, :] = 1

    st.session_state["필요인원표"] = 새


def 필요인원_dict_생성(표: pd.DataFrame, 조들):
    """DataFrame(조 x 요일)을 dict[요일][조] 형태로 변환."""
    return {요일: {조: int(표.loc[조, 요일]) for 조 in 조들} for 요일 in 요일들}


def 알바생저장_길이맞추기(조들):
    """
    st.session_state["알바생저장"]의 길이를 알바수에 맞추고,
    운영방식 변경 시 가능조/불가요일을 현재 조/요일에 맞게 보정.
    """
    저장 = st.session_state["알바생저장"]
    n = int(st.session_state["알바수"])

    # 부족하면 추가
    while len(저장) < n:
        저장.append({"이름": f"알바{len(저장)+1}", "가능조": 조들[:], "불가요일": []})

    # 초과하면 삭제
    if len(저장) > n:
        del 저장[n:]

    # 내용 보정(가능조/불가요일/이름)
    for i in range(len(저장)):
        이름 = (저장[i].get("이름", "") or "").strip()
        저장[i]["이름"] = 이름 or f"알바{i+1}"

        가능 = [x for x in 저장[i].get("가능조", []) if x in 조들]
        저장[i]["가능조"] = 가능 or 조들[:]

        불가 = [x for x in 저장[i].get("불가요일", []) if x in 요일들]
        저장[i]["불가요일"] = 불가

    st.session_state["알바생저장"] = 저장


def 알바생목록_생성(조들):
    """
    스케줄 생성용으로 가공된 알바 목록을 생성.
    - 가능조/불가요일은 set으로(검색/포함 검사 빠르게)
    - 이름 빈 값 방지
    """
    목록 = []
    for i, a in enumerate(st.session_state["알바생저장"]):
        이름 = (a.get("이름", "") or "").strip() or f"알바{i+1}"
        가능조 = set([x for x in a.get("가능조", []) if x in 조들]) or set(조들)
        불가요일 = set([x for x in a.get("불가요일", []) if x in 요일들])
        목록.append({"이름": 이름, "가능조": 가능조, "불가요일": 불가요일})
    return 목록


def 세션_정합성_동기화():
    """
    실행/불러오기 직후에도 꼬이지 않도록, 운영방식 기준으로
    (조들, 필요인원표, 필요인원 dict, 알바생저장)을 한 번에 정합 상태로 맞춘다.
    """
    조들 = 조리스트_가져오기(st.session_state["운영방식"])
    필요인원표_동기화(조들)
    st.session_state["필요인원"] = 필요인원_dict_생성(st.session_state["필요인원표"], 조들)
    알바생저장_길이맞추기(조들)
    return 조들


# ============================================================
# 3) 스케줄 생성 로직 (MRV + 공정 tie-break)
#    - MRV: 남은 슬롯 중 후보가 가장 적은 슬롯부터 채움(막힘 예방)
#    - tie-break: 배정횟수 적은 사람 우선 + 최근배정과 거리 고려
# ============================================================
def 스케줄_생성(알바생목록, 필요인원, 조들, seed=32, 같은날허용=False, 하루최대=1, 인접만=True):
    rng = random.Random(seed)

    결과 = {요일: {조: [] for 조 in 조들} for 요일 in 요일들}
    배정횟수 = {a["이름"]: 0 for a in 알바생목록}

    하루카운트 = {a["이름"]: {요일: 0 for 요일 in 요일들} for a in 알바생목록}
    최근배정요일 = {a["이름"]: -10**9 for a in 알바생목록}
    요일인덱스 = {요일: i for i, 요일 in enumerate(요일들)}
    조인덱스 = {조: i for i, 조 in enumerate(조들)}

    # 남은 슬롯 목록 구성
    남은슬롯 = []
    for 요일 in 요일들:
        for 조 in 조들:
            n = int(필요인원[요일][조])
            for k in range(n):
                남은슬롯.append((요일, 조, k))

    def 후보구하기(요일, 조):
        후보 = []
        for a in 알바생목록:
            이름 = a["이름"]

            # 개인 제약(불가 요일/가능 조)
            if 요일 in a["불가요일"]:
                continue
            if 조 not in a["가능조"]:
                continue

            # 같은 날 중복 근무 제한
            현재 = 하루카운트[이름][요일]
            if not 같은날허용:
                if 현재 >= 1:
                    continue
            else:
                if 현재 >= int(하루최대):
                    continue

                # 3조에서 같은 날 2타임이면 인접 조만 허용(오픈-미들/미들-마감)
                if 인접만 and len(조들) >= 3 and 현재 >= 1:
                    기존조들 = [다른조 for 다른조 in 조들 if 이름 in 결과[요일][다른조]]
                    if 기존조들:
                        기준 = 조인덱스[기존조들[0]]
                        if abs(조인덱스[조] - 기준) != 1:
                            continue

            후보.append(이름)
        return 후보

    while 남은슬롯:
        # MRV: 후보 수가 가장 적은 슬롯을 선택
        최솟값 = None
        후보정보 = None
        rng.shuffle(남은슬롯)  # 동률일 때 특정 슬롯에 쏠리지 않도록 섞음

        for 슬롯 in 남은슬롯:
            요일, 조, _ = 슬롯
            후보 = 후보구하기(요일, 조)
            c = len(후보)
            if 최솟값 is None or c < 최솟값:
                최솟값 = c
                후보정보 = (슬롯, 후보)
                if 최솟값 == 0:
                    break

        (요일, 조, k), 후보 = 후보정보

        # 후보가 0명이면 디버그 정보 제공
        if not 후보:
            탈락사유 = []
            for a in 알바생목록:
                이름 = a["이름"]
                사유 = []

                if 요일 in a["불가요일"]:
                    사유.append("불가요일")
                if 조 not in a["가능조"]:
                    사유.append("불가능조")

                현재 = 하루카운트[이름][요일]
                if not 같은날허용 and 현재 >= 1:
                    사유.append("같은날중복금지")
                if 같은날허용 and 현재 >= int(하루최대):
                    사유.append("하루최대초과")

                if 같은날허용 and 인접만 and len(조들) >= 3 and 현재 >= 1:
                    기존조들 = [다른조 for 다른조 in 조들 if 이름 in 결과[요일][다른조]]
                    if 기존조들:
                        기준 = 조인덱스[기존조들[0]]
                        if abs(조인덱스[조] - 기준) != 1:
                            사유.append("인접조만허용")

                if not 사유:
                    사유.append("알수없음")

                탈락사유.append({"이름": 이름, "사유": ", ".join(사유)})

            실패메시지 = f"배정 실패: '{요일} {조}' 슬롯(#{k+1})에 배정 가능한 사람이 없습니다."
            return None, 실패메시지, {"막힌슬롯": (요일, 조, k), "탈락사유": 탈락사유}

        # 공정 tie-break
        # 1) 배정횟수 적은 사람 우선
        # 2) 최근 배정과의 간격이 더 큰 사람 우선(연속 배정 방지 느낌)
        후보.sort(key=lambda name: (
            배정횟수[name],
            -(요일인덱스[요일] - 최근배정요일[name]),
        ))

        best0 = 후보[0]
        best_count = 배정횟수[best0]
        best_gap = (요일인덱스[요일] - 최근배정요일[best0])

        # 완전 동률이면 랜덤 선택
        최적후보 = [
            name for name in 후보
            if 배정횟수[name] == best_count and (요일인덱스[요일] - 최근배정요일[name]) == best_gap
        ]
        선택 = rng.choice(최적후보)

        # 배정 반영
        결과[요일][조].append(선택)
        배정횟수[선택] += 1
        하루카운트[선택][요일] += 1
        최근배정요일[선택] = 요일인덱스[요일]

        남은슬롯.remove((요일, 조, k))

    return (결과, 배정횟수), None, None


# ============================================================
# 4) UI 구성
# ============================================================
왼쪽, 오른쪽 = st.columns([1, 1.2])

with 왼쪽:
    # --------------------------------------------------------
    # 4-1) 운영방식
    # --------------------------------------------------------
    st.subheader("1) 업장 운영 방식")
    운영방식 = st.selectbox(
        "운영 방식 선택",
        ["3조(오픈/미들/마감)", "2조(오픈/마감)", "1인(하루 1명)"],
        index=["3조(오픈/미들/마감)", "2조(오픈/마감)", "1인(하루 1명)"].index(st.session_state["운영방식"])
    )
    st.session_state["운영방식"] = 운영방식
    조들 = 세션_정합성_동기화()

    # --------------------------------------------------------
    # 4-2) 필요 인원 표 (form 적용 방식)
    # --------------------------------------------------------
    st.subheader("2) 요일/조별 필요 인원 (적용 버튼 누를 때만 반영)")
    with st.form("필요인원_폼", clear_on_submit=False):
        임시표 = st.session_state["필요인원표"].copy()

        c1, c2, c3 = st.columns(3)
        전체1 = c1.form_submit_button("전체 1로 채우기", use_container_width=True)
        전체0 = c2.form_submit_button("전체 0으로 채우기", use_container_width=True)
        평주 = c3.form_submit_button("평일 1 / 주말 2", use_container_width=True)

        if 전체1:
            임시표.loc[:, :] = 1
        if 전체0:
            임시표.loc[:, :] = 0
        if 평주:
            임시표.loc[:, 평일] = 1
            임시표.loc[:, 주말] = 2

        편집값 = st.data_editor(임시표, use_container_width=True, num_rows="fixed", key="필요인원_에디터")
        적용 = st.form_submit_button("필요 인원 적용", type="primary", use_container_width=True)

        if 적용:
            편집값 = 편집값.fillna(0)
            for 조 in 조들:
                for 요일 in 요일들:
                    try:
                        편집값.loc[조, 요일] = int(편집값.loc[조, 요일])
                    except Exception:
                        편집값.loc[조, 요일] = 0

            st.session_state["필요인원표"] = 편집값.astype("int64")
            st.session_state["필요인원"] = 필요인원_dict_생성(st.session_state["필요인원표"], 조들)
            st.success("필요 인원을 적용했습니다.")

    # 항상 최신 세션값 사용
    필요인원 = st.session_state["필요인원"]

    # --------------------------------------------------------
    # 4-3) 같은 날 중복 옵션
    # --------------------------------------------------------
    st.subheader("3) 같은 날 중복 근무 옵션")
    같은날허용 = st.checkbox("같은 날 2타임 이상 허용", value=st.session_state["옵션_같은날허용"])
    st.session_state["옵션_같은날허용"] = 같은날허용

    if 같은날허용:
        하루최대 = st.number_input("1인 하루 최대 근무 횟수", min_value=1, max_value=3,
                               value=int(st.session_state["옵션_하루최대"]), step=1)
        st.session_state["옵션_하루최대"] = int(하루최대)

        인접만 = st.checkbox("3조일 때, 같은 날 2타임은 인접(오픈-미들/미들-마감)만 허용",
                         value=st.session_state["옵션_인접만"])
        st.session_state["옵션_인접만"] = 인접만
    else:
        st.session_state["옵션_하루최대"] = 1
        하루최대 = 1
        인접만 = False

    # --------------------------------------------------------
    # 4-4) 알바생 명단 (form 적용 방식)
    # --------------------------------------------------------
    st.subheader("4) 알바생 명단 (적용 버튼 누를 때만 반영)")

    a1, a2, a3 = st.columns([1, 1, 3])
    with a1:
        if st.button("➖ 1명", use_container_width=True):
            st.session_state["알바수"] = max(1, int(st.session_state["알바수"]) - 1)
            알바생저장_길이맞추기(조들)
            st.session_state["알바폼버전"] += 1
            st.rerun()

    with a2:
        if st.button("➕ 1명", use_container_width=True):
            st.session_state["알바수"] = min(30, int(st.session_state["알바수"]) + 1)
            알바생저장_길이맞추기(조들)
            st.session_state["알바폼버전"] += 1
            st.rerun()

    with a3:
        st.markdown(f"**현재 알바생 수: {int(st.session_state['알바수'])}명**")

    # 폼 내부에서만 편집 -> 적용 눌렀을 때만 확정 저장
    with st.form("알바생_폼", clear_on_submit=False):
        v = st.session_state["알바폼버전"]
        임시저장 = st.session_state["알바생저장"]

        for idx in range(int(st.session_state["알바수"])):
            st.markdown(f"**알바 {idx+1}**")

            저장 = 임시저장[idx]
            기본이름 = 저장.get("이름", f"알바{idx+1}")
            기본가능 = [x for x in 저장.get("가능조", []) if x in 조들] or 조들[:]
            기본불가 = [x for x in 저장.get("불가요일", []) if x in 요일들]

            st.text_input("이름", value=기본이름, key=f"v{v}_name_{idx}")
            st.multiselect("가능 조", options=조들, default=기본가능, key=f"v{v}_can_{idx}")
            st.multiselect("불가 요일", options=요일들, default=기본불가, key=f"v{v}_cant_{idx}")

        알바적용 = st.form_submit_button("알바생 명단 적용", type="primary", use_container_width=True)

        if 알바적용:
            새알바 = []
            for idx in range(int(st.session_state["알바수"])):
                이름 = (st.session_state.get(f"v{v}_name_{idx}", "") or "").strip()
                가능조 = list(st.session_state.get(f"v{v}_can_{idx}", []))
                불가요일 = list(st.session_state.get(f"v{v}_cant_{idx}", []))

                # 빈 값/비정합 값 방지
                if not 이름:
                    이름 = f"알바{idx+1}"
                가능조 = [x for x in 가능조 if x in 조들] or 조들[:]
                불가요일 = [x for x in 불가요일 if x in 요일들]

                새알바.append({"이름": 이름, "가능조": 가능조, "불가요일": 불가요일})

            st.session_state["알바생저장"] = 새알바
            st.session_state["알바폼버전"] += 1
            st.success("알바생 명단을 적용했습니다. (JSON 저장에도 이 값이 사용됩니다.)")
            st.rerun()

    st.caption("※ 스케줄 생성은 ‘마지막으로 적용된 알바생 명단’을 기준으로 수행됩니다.")

    # --------------------------------------------------------
    # 4-5) JSON 저장/불러오기
    # --------------------------------------------------------
    st.subheader("5) 명단 저장/불러오기")

    설정데이터 = {
        "운영방식": st.session_state["운영방식"],
        "필요인원표": st.session_state["필요인원표"].to_dict(),
        "옵션": {
            "같은날허용": st.session_state["옵션_같은날허용"],
            "하루최대": st.session_state["옵션_하루최대"],
            "인접만": st.session_state["옵션_인접만"],
        },
        "알바생": st.session_state["알바생저장"],
    }

    json_bytes = json.dumps(설정데이터, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("현재 설정 명단 다운로드", data=json_bytes,
                       file_name="scheduler_settings.json", mime="application/json")

    업로드 = st.file_uploader("명단 불러오기", type=["json"], key="json_uploader")
    if 업로드 is not None:
        try:
            파일바이트 = 업로드.getvalue()
            파일해시 = hashlib.md5(파일바이트).hexdigest()

            # 같은 파일은 1번만 적용(무한 rerun 방지)
            if st.session_state.get("마지막_json_해시") != 파일해시:
                로드 = json.loads(파일바이트.decode("utf-8"))

                # 운영방식 먼저 적용
                st.session_state["운영방식"] = 로드.get("운영방식", st.session_state["운영방식"])
                조들_로드 = 조리스트_가져오기(st.session_state["운영방식"])

                # 필요인원표 적용(방향/형태 불일치 대비 포함)
                필요인원표_동기화(조들_로드)
                표dict = 로드.get("필요인원표", None)

                if 표dict is not None:
                    df_loaded = pd.DataFrame(표dict)

                    if set(df_loaded.columns) == set(요일들) and set(df_loaded.index) == set(조들_로드):
                        pass
                    elif set(df_loaded.columns) == set(조들_로드) and set(df_loaded.index) == set(요일들):
                        df_loaded = df_loaded.T
                    else:
                        tmp = pd.DataFrame(0, index=조들_로드, columns=요일들, dtype="int64")
                        for 조 in 조들_로드:
                            for 요일 in 요일들:
                                if 조 in df_loaded.index and 요일 in df_loaded.columns:
                                    try:
                                        tmp.loc[조, 요일] = int(df_loaded.loc[조, 요일])
                                    except Exception:
                                        tmp.loc[조, 요일] = 0
                        df_loaded = tmp

                    df_loaded = df_loaded.fillna(0).astype("int64")
                    st.session_state["필요인원표"] = df_loaded

                # 옵션 적용
                옵션 = 로드.get("옵션", {})
                st.session_state["옵션_같은날허용"] = bool(옵션.get("같은날허용", st.session_state["옵션_같은날허용"]))
                st.session_state["옵션_하루최대"] = int(옵션.get("하루최대", st.session_state["옵션_하루최대"]))
                st.session_state["옵션_인접만"] = bool(옵션.get("인접만", st.session_state["옵션_인접만"]))

                # 알바 적용
                알바 = 로드.get("알바생", [])
                if isinstance(알바, list) and len(알바) >= 1:
                    st.session_state["알바생저장"] = 알바
                    st.session_state["알바수"] = len(알바)

                # 최종 정합성 보정(운영방식 기준)
                알바생저장_길이맞추기(조들_로드)
                st.session_state["필요인원"] = 필요인원_dict_생성(st.session_state["필요인원표"], 조들_로드)

                # 폼 키 갱신 + 처리 완료 기록
                st.session_state["알바폼버전"] += 1
                st.session_state["마지막_json_해시"] = 파일해시

                st.success("명단을 불러왔습니다. (1회 적용 완료)")
                st.rerun()
            else:
                st.info("이미 이 명단을 적용한 상태입니다.")
        except Exception as e:
            st.error(f"명단 불러오기 실패: {e}")


with 오른쪽:
    # --------------------------------------------------------
    # 4-6) 결과 / 스케줄 생성
    # --------------------------------------------------------
    st.subheader("결과")
    seed = st.number_input("시드", min_value=0, max_value=10_000_000, value=32, step=1)
    생성 = st.button("스케줄 생성", type="primary")

    if 생성:
        # 생성 시점에도 “세션 기준으로” 재계산(불러오기 직후에도 100% 일치)
        조들_실행 = 조리스트_가져오기(st.session_state["운영방식"])
        필요인원표_동기화(조들_실행)
        st.session_state["필요인원"] = 필요인원_dict_생성(st.session_state["필요인원표"], 조들_실행)
        필요인원_실행 = st.session_state["필요인원"]

        알바생저장_길이맞추기(조들_실행)
        알바생목록_실행 = 알바생목록_생성(조들_실행)

        # (선택) 생성 전 간단 요약
        총슬롯 = sum(int(필요인원_실행[요일][조]) for 요일 in 요일들 for 조 in 조들_실행)
        st.info(f"운영방식: {st.session_state['운영방식']} | 알바수: {len(알바생목록_실행)} | 총 슬롯: {총슬롯}")

        결과_배정, 에러, 디버그 = 스케줄_생성(
            알바생목록_실행,
            필요인원_실행,
            조들_실행,
            seed=int(seed),
            같은날허용=st.session_state["옵션_같은날허용"],
            하루최대=int(st.session_state["옵션_하루최대"]),
            인접만=st.session_state["옵션_인접만"],
        )

        if 에러:
            st.error(에러)
            if 디버그 is not None:
                요일, 조, k = 디버그["막힌슬롯"]
                st.markdown(f"#### 막힌 슬롯: **{요일} {조} (#{k+1})**")
                df_debug = pd.DataFrame(디버그["탈락사유"])
                st.dataframe(df_debug, use_container_width=True, hide_index=True)
        else:
            결과, 배정횟수 = 결과_배정

            # 결과 표(조 x 요일)
            표 = []
            for 조 in 조들_실행:
                row = {"조": 조}
                for 요일 in 요일들:
                    row[요일] = ", ".join(결과[요일][조]) if 결과[요일][조] else "-"
                표.append(row)
            df = pd.DataFrame(표)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # 배정 횟수
            st.markdown("### 알바별 배정 횟수")
            df2 = (
                pd.DataFrame([{"이름": k, "배정횟수": v} for k, v in 배정횟수.items()])
                .sort_values(["배정횟수", "이름"])
                .reset_index(drop=True)
            )
            st.dataframe(df2, use_container_width=True, hide_index=True)

            최소배정 = df2["배정횟수"].min()
            최대배정 = df2["배정횟수"].max()
            st.info(f"총 슬롯: {총슬롯} | 최소 배정: {최소배정} | 최대 배정: {최대배정} | 편차: {최대배정-최소배정}")

            # CSV 다운로드
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("엑셀표 다운로드", data=csv, file_name="schedule.csv", mime="text/csv")
