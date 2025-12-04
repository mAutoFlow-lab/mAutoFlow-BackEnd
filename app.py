# app.py - mAutoFlow 백엔드 전용

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import os
from jose import jwt, JWTError

from c_autodiag import extract_function_body, StructuredFlowEmitter, extract_function_names

app = FastAPI()

# --- Supabase JWT 설정 ---
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")
JWT_ALG = "HS256"  # Supabase 기본 알고리즘

def verify_access_token(access_token: str):
    """
    프론트에서 넘어온 access_token 이 진짜 Supabase 가 발급한 토큰인지 확인.
    성공하면 payload(claims)를 리턴, 실패하면 HTTP 401 에러.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing access_token")

    if not SUPABASE_JWT_SECRET:
        # Render Environment 에 값이 안 들어가 있으면 500
        raise HTTPException(status_code=500, detail="SUPABASE_JWT_SECRET is not configured")

    try:
        payload = jwt.decode(
            access_token,
            SUPABASE_JWT_SECRET,
            algorithms=[JWT_ALG],
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid access_token: {e}")


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
    # 1) 토큰 검증 (실패하면 여기서 401 에러 발생)
    user_claims = verify_access_token(access_token)

    # (선택) 유저 정보 꺼내 쓰고 싶으면 여기서 꺼내면 됨
    user_id = user_claims.get("sub")
    user_email = user_claims.get("email")

    # TODO: 여기서부터 사용량/요금제 로직 넣을 수 있음
    #   - 예: 이메일에 따라 무료/유료 플랜 구분
    #   - 예: DB에 오늘 사용 횟수 저장/체크 등

    try:
        mermaid, func_name, node_lines = generate_mermaid_auto(
            source_code,
            branch_shape=branch_shape
        )
        return JSONResponse(
            {
                "mermaid": mermaid,
                "func_name": func_name,
                "node_lines": node_lines,
            }
        )
    except Exception as e:
        return JSONResponse({"mermaid": "", "func_name": "", "error": str(e)})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
