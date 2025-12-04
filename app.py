# app.py - mAutoFlow 백엔드 전용

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import date
from collections import defaultdict

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


DAILY_FREE_LIMIT = 5
FREE_NODE_LIMIT = 20

_usage_counter = defaultdict(lambda: {"date": date.today(), "count": 0})

def check_daily_limit(user_id: str):
    today = date.today()
    info = _usage_counter[user_id]

    # 날짜 바뀌면 초기화
    if info["date"] != today:
        info["date"] = today
        info["count"] = 0

    # 무료 제한 체크
    if info["count"] >= DAILY_FREE_LIMIT:
        raise HTTPException(status_code=429, detail="DAILY_LIMIT_EXCEEDED")

    # 정상 → 1 증가
    info["count"] += 1


def generate_mermaid_auto(source_code: str, branch_shape: str = "rounded"):
    func_list = extract_function_names(source_code)
    if not func_list:
        raise ValueError("The function could not be found in the code.")
    func_name = "main" if "main" in func_list else func_list[0]

    body = extract_function_body(source_code, func_name)

    body_index = source_code.find(body)
    if body_index == -1:
        body_start_line = 0
    else:
        body_start_line = source_code[:body_index].count("\n")

    emitter = StructuredFlowEmitter(func_name, branch_shape=branch_shape)
    mermaid = emitter.emit_from_body(body)

    node_lines = {
        nid: body_start_line + line_idx
        for nid, line_idx in emitter.node_line_map.items()
    }

    return mermaid, func_name, node_lines


@app.get("/")
async def health():
    return {"status": "ok", "service": "mAutoFlow backend"}


@app.post("/api/convert_text")
async def convert_c_text_to_mermaid(
    source_code: str = Form(...),
    branch_shape: str = Form("rounded"),
    access_token: str = Form(None),   # 👈 프론트에서 보내는 토큰
):
    # 1) 토큰 검증
    user_claims = verify_access_token(access_token)

    # 여기서 user_id 를 하나 정해줘야 함
    # 지금 verify_access_token 이 {"token": access_token} 만 돌려주니까
    # 일단은 토큰 문자열 자체를 user_id 로 써도 됨.
    user_id = user_claims["token"]

    # ✅ 하루 무료 사용량 체크 (백엔드 레벨)
    check_daily_limit(user_id)

    # (나중에 JWT decode 를 다시 붙이면)
    # user_id = payload["sub"]  같은 걸로 바꾸면 됨.

    try:
        mermaid, func_name, node_lines = generate_mermaid_auto(
            source_code,
            branch_shape=branch_shape
        )

        # 노드 수 제한 (백엔드에서도 한 번 더)
        node_count = len(node_lines)
        if node_count > FREE_NODE_LIMIT:
            return JSONResponse(
                status_code=400,
                content={
                    "mermaid": "",
                    "func_name": "",
                    "error": "TOO_MANY_NODES",
                    "error_code": "TOO_MANY_NODES",
                },
            )
        
        return JSONResponse(
            {
                "mermaid": mermaid,
                "func_name": func_name,
                "node_lines": node_lines,
            }
        )
    except HTTPException:
        # check_daily_limit 에서 던진 건 그대로 통과
        raise
    except Exception as e:
        return JSONResponse({"mermaid": "", "func_name": "", "error": str(e)})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
