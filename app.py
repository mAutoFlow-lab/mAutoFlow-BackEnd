# app.py - mAutoFlow 백엔드 전용

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import date, datetime
from collections import defaultdict

import hashlib
import datetime as dt
import re
# import os
# from jose import jwt, JWTError

from c_autodiag import extract_function_body, StructuredFlowEmitter, extract_function_names

app = FastAPI()

def verify_access_token(access_token: str | None):
    """
    일단은 '로그인해서 토큰을 보내고 있는지' 정도만 확인.
    토큰 서명 검증은 나중에 Supabase 설정이 안정되면 다시 추가.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing access_token")
    # 나중에 여기에 jwt.decode(...)를 다시 넣으면 됨
    return {"token": access_token}


# CORS: 프론트 도메인(.netlify.app)을 넣어준다.
# 개발 중에는 "*" 로 열어둬도 되고, 상용에서는 꼭 도메인으로 제한하자.
origins = [
    "https://mautoflow-frontend.pages.dev",  # 새 Cloudflare 프론트
    "http://localhost:8000",
    "https://mautoflow-lab.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 개발 단계에서는 ["*"] 도 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEPLOY_VERSION = "v0.0.3"
DAILY_FREE_LIMIT = 5
FREE_NODE_LIMIT = 20

# user_id 별로 오늘 날짜, 사용 횟수, 마지막 코드 해시를 기억
_usage_counter = defaultdict(
    lambda: {"date": date.today(), "count": 0, "last_code_hash": None}
)

def normalize_source(code: str) -> str:
    """
    같은 함수인데 공백만 조금 바뀐 경우는 동일 코드로 취급하기 위해
    라인 끝 공백을 제거하고 앞뒤 공백을 정리한다.
    """
    lines = code.strip().splitlines()
    lines = [ln.rstrip() for ln in lines]
    return "\n".join(lines)


def make_code_hash(code: str) -> str:
    norm = normalize_source(code)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def extract_full_function_signature(source_code: str, func_name: str) -> str:
    """
    소스 코드 전체에서 해당 함수의 '선언부'를 최대한 그대로 찾아서 반환.
    예)
      static int Foo(int a, int b)
    이런 식으로 리턴타입 + 이름 + 인자까지 포함된 한 줄(또는 멀티라인)을 정리해서 리턴.
    """
    # 멀티라인 함수 선언도 잡기 위해, 줄바꿈을 공백으로 한 번 눌러서 찾는다.
    # (너무 복잡하게 안 가고, 일단 실용적인 수준으로만)
    code_one_line = re.sub(r"\s+", " ", source_code)

    # AUTOSAR FUNC(...) 도 대충 지원
    pattern = re.compile(
        r"""
        (                               # 전체 시그니처 캡쳐
            (?:FUNC\s*\([^)]*\)\s*)?     #   AUTOSAR FUNC(...) (옵션)
            [A-Za-z_][\w\s\*\(\)]*?      #   리턴타입/수식어(대충)
            \b""" + re.escape(func_name) + r"""\s*  #   함수 이름
            \(
                [^)]*
            \)
        )
        """,
        re.VERBOSE,
    )

    m = pattern.search(code_one_line)
    if not m:
        # 못 찾으면 fallback: func_name()
        return f"{func_name}()"

    sig = m.group(1).strip()

    # 공백 정리
    sig = re.sub(r"\s+", " ", sig).strip()

    return sig


def check_daily_limit(user_id: str, code_hash: str) -> int:
    """
    - user_id 기준으로 오늘 날짜의 사용량을 관리한다.
    - 같은 코드(code_hash)가 들어오면 count 를 증가시키지 않는다.
    - 다른 코드가 들어왔고, 이미 DAILY_FREE_LIMIT 만큼 썼다면 429를 던진다.
    """
    today = date.today()
    info = _usage_counter[user_id]

    # 날짜가 바뀌면 카운터 리셋
    if info["date"] != today:
        info["date"] = today
        info["count"] = 0
        info["last_code_hash"] = None

    last_hash = info.get("last_code_hash")
    is_new_code = (last_hash != code_hash)

    # 새로운 코드인데, 이미 한도까지 사용한 경우에만 막는다
    if is_new_code and info["count"] >= DAILY_FREE_LIMIT:
        print(
            f"[USAGE] LIMIT_EXCEEDED user_id={user_id} "
            f"date={info['date']} count={info['count']}"
        )
        raise HTTPException(
            status_code=429,
            detail={
                "code": "DAILY_LIMIT_EXCEEDED",
                "usage_count": info["count"],
                "daily_free_limit": DAILY_FREE_LIMIT,
            },
        )

    # 새로운 코드면 +1, 같은 코드면 카운트 유지
    if is_new_code:
        info["count"] += 1
        info["last_code_hash"] = code_hash
        print(
            f"[USAGE] OK (new code) user_id={user_id} "
            f"date={info['date']} count={info['count']}"
        )
    else:
        print(
            f"[USAGE] OK (same code) user_id={user_id} "
            f"date={info['date']} count={info['count']}"
        )

    return info["count"]


def generate_mermaid_auto(source_code: str, branch_shape: str = "rounded"):
    """
    1) 코드에서 함수 목록을 전부 찾는다.
    2) main이 있으면 main, 없으면 첫 번째 함수를 우선 시도한다.
    3) 선택한 함수에서 본문 추출이 실패하면, 나머지 함수들까지 순차적으로 시도.
    4) 결국 아무 함수도 본문 추출이 안 되면, 어떤 함수들을 발견했는지까지 에러 메시지에 포함.
    """
    # 1) 함수 목록 탐색
    func_list = extract_function_names(source_code)
    print("[DEBUG] detected functions:", func_list)  # 디버그용

    if not func_list:
        # 아예 함수 정의를 찾지 못한 경우
        raise ValueError(
            "The function could not be found in the code. "
            "Check that you pasted the full function definition (including its header with '{')."
        )

    # 2) 우선 시도할 함수 선택
    preferred = "main" if "main" in func_list else func_list[0]

    tried = []
    last_err = None
    body = None
    func_name = None

    # 3) 우선 함수 + 나머지 함수들 순서대로 시도
    for name in [preferred] + [f for f in func_list if f != preferred]:
        tried.append(name)
        try:
            body = extract_function_body(source_code, name)
            func_name = name
            break
        except Exception as e:
            last_err = e
            continue

    if body is None or func_name is None:
        # 어떤 함수에서도 본문을 못 뽑은 경우
        msg = f"Failed to extract function body. Tried: {tried}"
        if last_err is not None:
            msg += f" | Last error: {last_err}"
        raise ValueError(msg)

    # 4) 여기부터는 기존 로직 그대로
    body_index = source_code.find(body)
    if body_index == -1:
        body_start_line = 0
    else:
        body_start_line = source_code[:body_index].count("\n")

    # 여기서 풀 시그니처 생성
    full_signature = extract_full_function_signature(source_code, func_name)

    emitter = StructuredFlowEmitter(func_name, branch_shape=branch_shape)
    mermaid = emitter.emit_from_body(body)

    node_lines = {
        nid: body_start_line + line_idx
        for nid, line_idx in emitter.node_line_map.items()
    }

    # full_signature 를 함께 리턴
    return mermaid, func_name, node_lines, full_signature


@app.get("/version")
async def version():
    """
    프론트에서 백엔드 버전/제한 값을 확인할 수 있는 엔드포인트
    """
    return {
        "service": "mAutoFlow backend",
        "version": DEPLOY_VERSION,
        "daily_free_limit": DAILY_FREE_LIMIT,
        "free_node_limit": FREE_NODE_LIMIT,
        "server_time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/")
async def health():
    return {"status": "ok", "service": "mAutoFlow backend"}


@app.post("/api/convert_text")
async def convert_c_text_to_mermaid(
    source_code: str = Form(...),
    branch_shape: str = Form("rounded"),
    access_token: str = Form(None),
    user_id: str | None = Form(None),
    user_email: str | None = Form(None),
):
    verify_access_token(access_token)

    print(f"[REQ] /api/convert_text user_id={user_id} email={user_email}")

    if not user_id:
        raise HTTPException(status_code=400, detail="MISSING_USER_ID")

    usage_count: int | None = None

    # 같은 코드면 사용 횟수를 올리지 않기 위해 해시를 만든다
    code_hash = make_code_hash(source_code)

    # 🔹 테스트 계정 여부 플래그
    is_test_account = (user_email == "exitgiveme@gmail.com")

    # 테스트 계정은 일일 회수 제한도 건너뛴다
    if is_test_account:
        print("[API] test account, no daily limit / no node limit")
    else:
        # 코드 해시를 기준으로, "새로운 코드"일 때만 사용량 증가
        usage_count = check_daily_limit(user_id, code_hash)

    try:
        mermaid, func_name, node_lines, full_signature = generate_mermaid_auto(
            source_code,
            branch_shape=branch_shape,
        )

        node_count = len(node_lines)

        # 일반 유저만 노드 제한 적용, 테스트 계정은 무제한
        if (not is_test_account) and node_count > FREE_NODE_LIMIT:
            return JSONResponse(
                status_code=400,
                content={
                    "mermaid": "",
                    "func_name": "",
                    "error": "TOO_MANY_NODES",
                    "error_code": "TOO_MANY_NODES",
                    # 사용량 정보도 같이 내려주고 싶으면 여기서 usage_count 포함 가능
                    "usage_count": usage_count,
                    "daily_free_limit": DAILY_FREE_LIMIT,
                    "free_node_limit": FREE_NODE_LIMIT,
                },
            )

        return JSONResponse(
            {
                "mermaid": mermaid,
                "func_name": func_name,
                "full_signature": full_signature,   # 추가
                "node_lines": node_lines,
                "usage_count": usage_count,
                "daily_free_limit": DAILY_FREE_LIMIT,
                "free_node_limit": FREE_NODE_LIMIT,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            {
                "mermaid": "",
                "func_name": "",
                "error": str(e),
            }
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
