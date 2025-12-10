# app.py - mAutoFlow 백엔드 전용

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import date, datetime
from collections import defaultdict

import os
import hashlib
import hmac        # webhook 검증용
import datetime as dt
import re
import requests
from supabase import create_client, Client
from jose import jwt, JWTError

from c_autodiag import extract_function_body, StructuredFlowEmitter, extract_function_names

app = FastAPI()

# --- Supabase client 설정 ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

supabase: Client | None = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
else:
    # 로컬 테스트나 설정 실수 시 바로 죽지 말고 로그만 남김
    # (완전 엄격하게 하고 싶으면 여기서 RuntimeError를 다시 써도 됨)
    print("[WARN] SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY가 설정되지 않았습니다.")
# --- 여기까지 ---


@app.post("/webhook/lemon")
async def lemon_webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Signature")

    secret = os.getenv("LEMON_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Missing webhook secret")

    # 1) 시그니처 검증
    expected_sig = hmac.new(
        secret.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature or "", expected_sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # 2) payload / event 추출
    payload = await request.json()
    event = payload.get("meta", {}).get("event_name")

    # 3) supabase 클라이언트 확보
    db = get_supabase_client()

    # 4) 공통으로 쓸 subscription 데이터 파싱
    sub = payload.get("data", {}) or {}
    attrs = sub.get("attributes", {}) or {}

    lemon_subscription_id = sub.get("id")
    if not lemon_subscription_id:
        raise HTTPException(status_code=400, detail="Missing subscription id")

    # ✅ Lemon 쪽에서 넘어온 구독자 이메일
    user_email = attrs.get("user_email")

    # ✅ 이메일로 Supabase user_id 찾기 (profiles 테이블 기준)
    user_id = lookup_user_id_by_email(db, user_email)

    base_row = {
        # 🔑 Supabase 유저와의 연결
        "user_id": user_id,

        "lemon_subscription_id": lemon_subscription_id,
        "lemon_customer_id": attrs.get("customer_id"),
        "lemon_order_id": attrs.get("order_id"),
        "product_id": attrs.get("product_id"),
        "variant_id": attrs.get("variant_id"),
        "plan_name": attrs.get("product_name") or attrs.get("variant_name"),
        "status": attrs.get("status"),
        "is_trial": attrs.get("status") == "on_trial",
        "trial_ends_at": attrs.get("trial_ends_at"),
        "renews_at": attrs.get("renews_at"),
        "ends_at": attrs.get("ends_at"),
    }

    # ---------------------------------------------------
    #  Subscription Created
    # ---------------------------------------------------
    if event == "subscription_created":
        try:
            db.table("subscriptions").insert(base_row).execute()
        except Exception as e:
            msg = str(e)
            if "duplicate key value violates unique constraint" in msg:
                print("[WEBHOOK] duplicate subscription_created, ignore:", msg)
            else:
                raise

    # ---------------------------------------------------
    #  Subscription Updated
    # ---------------------------------------------------
    elif event == "subscription_updated":
        # ⚠ 이미 DB에 user_id가 있을 수도 있으니, None 으로 덮어쓰지 않게 처리
        update_row = dict(base_row)
        if user_id is None:
            update_row.pop("user_id", None)

        db.table("subscriptions").update(update_row) \
          .eq("lemon_subscription_id", lemon_subscription_id) \
          .execute()

    # ---------------------------------------------------
    # Subscription Cancelled
    # ---------------------------------------------------
    elif event == "subscription_cancelled":
        cancel_row = dict(base_row)
        cancel_row["status"] = "cancelled"
        if user_id is None:
            cancel_row.pop("user_id", None)

        db.table("subscriptions").update(cancel_row) \
          .eq("lemon_subscription_id", lemon_subscription_id) \
          .execute()

    else:
        print(f"[WEBHOOK] ignore event: {event}")

    return {"ok": True}



def get_user_subscription(user_id: str | None):
    """
    주어진 user_id 에 대한 'active' 구독 한 개를 반환한다.
    없으면 None.
    """
    if not user_id:
        return None

    if supabase is None:
        print("[SUBS] supabase client not initialized")
        return None

    try:
        resp = (
            supabase
            .table("subscriptions")
            .select("plan_name,plan_tier,status,is_trial,trial_ends_at,renews_at,ends_at")
            .eq("user_id", user_id)
            .eq("status", "active")   # active 플랜만 본다
            .limit(1)
            .execute()
        )
    except Exception as e:
        print("[SUBS] error querying subscriptions:", e)
        return None

    rows = getattr(resp, "data", None) or []
    if not rows:
        print("[SUBS] no active subscription for user:", user_id)
        return None

    row = rows[0]
    print("[SUBS] active subscription:", row)
    return row


def lookup_user_id_by_email(db: Client, email: str | None) -> str | None:
    """
    Lemon 구독 webhook payload 안의 user_email 로
    Supabase 쪽 user_id 를 찾는다.
    - 여기서는 public.profiles 테이블에 (id, email) 이 있다고 가정.
      만약 테이블 이름이 다르면 아래 table("profiles") 부분만 수정.
    """
    if not email:
        return None

    try:
        resp = (
            db.table("profiles")  # <-- 너희 실제 유저 테이블 이름으로 변경 가능
              .select("id")
              .eq("email", email)
              .limit(1)
              .execute()
        )
    except Exception as e:
        print("[WEBHOOK] lookup_user_id_by_email error:", e)
        return None

    rows = getattr(resp, "data", None) or []
    if not rows:
        print("[WEBHOOK] no matching user for email:", email)
        return None

    user_id = rows[0].get("id")
    print(f"[WEBHOOK] matched email {email} -> user_id {user_id}")
    return user_id



def verify_access_token(access_token: str | None):
    """
    Supabase JWT(access_token)를 검증해서 user_id, email을 꺼낸다.
    - 서명이 틀리거나 만료되면 401 에러
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing access_token")

    if not SUPABASE_JWT_SECRET:
        # 설정 안 되어 있으면 그냥 존재만 체크하고 통과 (임시 fallback)
        print("[AUTH] WARNING: SUPABASE_JWT_SECRET not set, skipping JWT verify")
        return {"user_id": None, "email": None}

    try:
        # Supabase 기본 설정은 HS256 + audience "authenticated"
        payload = jwt.decode(
            access_token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
        )

        user_id = payload.get("sub")
        email = payload.get("email")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no sub")

        return {"user_id": user_id, "email": email}

    except JWTError as e:
        print("[AUTH] JWT decode error:", e)
        raise HTTPException(status_code=401, detail="Invalid access_token")


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
FREE_NODE_LIMIT    = 20       # Free: 20 nodes
PRO_NODE_LIMIT     = 200      # Pro : 200 nodes
EXPERT_NODE_LIMIT  = 1000     # Expert: 1000 nodes

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
    소스 코드 전체에서 해당 함수의 '선언부'를 최대한 찾아서 반환.
    - AUTOSAR FUNC(...)인 경우: FUNC(...) + 이름 + 매개변수 전체
    - 일반 C 함수인 경우: static/inline/반환타입까지 포함한 시그니처
    """
    # 모든 공백/줄바꿈을 하나의 공백으로 눌러서 단일 문자열로 만든다.
    flat = re.sub(r"\s+", " ", source_code)

    # func_name( 위치 찾기
    pattern = r"\b" + re.escape(func_name) + r"\s*\("
    m = re.search(pattern, flat)
    if not m:
        return f"{func_name}()"

    # 함수 이름 뒤의 '(' 위치
    paren_start = flat.find("(", m.start())
    if paren_start == -1:
        return f"{func_name}()"

    # 괄호 깊이 카운트하면서, 매개변수 리스트의 마지막 ')' 위치 찾기
    depth = 0
    end_idx = None
    for i in range(paren_start, len(flat)):
        ch = flat[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx is None:
        # 균형이 맞지 않으면 fallback
        return f"{func_name}()"

    # AUTOSAR FUNC(...) 포함을 위해, 함수 이름 앞쪽에서 FUNC( 를 찾아본다.
    search_window_start = max(0, m.start() - 200)  # 뒤로 200자 정도만 본다.
    window = flat[search_window_start:m.start()]
    macro_pos = window.rfind("FUNC(")

    # 1) AUTOSAR FUNC(...) 패턴: FUNC(...) 부터 끝까지 사용
    if macro_pos != -1:
        sig_start = search_window_start + macro_pos
        sig = flat[sig_start:end_idx + 1].strip()
        return sig

    # 2) 일반 C 함수: 반환 타입까지 포함한 시그니처를 정규식으로 시도
    m2 = re.search(
        r"([A-Za-z_][\w\s\*\(\)]*\b" + re.escape(func_name) + r"\s*\([^)]*\))",
        flat,
    )
    if m2:
        return m2.group(1).strip()

    # 3) 그래도 못 찾으면, 최소한 이름+인자만이라도 반환
    sig = flat[m.start():end_idx + 1].strip()
    return sig


def _check_daily_limit_memory(user_id: str, code_hash: str) -> int:
    """
    (백업용) 메모리 기반 일일 사용량 카운터.
    - Supabase DB를 사용할 수 없을 때만 fallback 으로 사용한다.
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
            f"[USAGE-MEM] LIMIT_EXCEEDED user_id={user_id} "
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
            f"[USAGE-MEM] OK (new code) user_id={user_id} "
            f"date={info['date']} count={info['count']}"
        )
    else:
        print(
            f"[USAGE-MEM] OK (same code) user_id={user_id} "
            f"date={info['date']} count={info['count']}"
        )

    return info["count"]

def check_daily_limit(user_id: str, code_hash: str) -> int:
    """
    Supabase public.diagram_usage 테이블을 사용해서
    무료 플랜(free) 유저의 일일 사용량을 관리한다.

    - user_id + usage_date 기준으로 한 줄만 유지.
    - 같은 코드(code_hash)가 다시 들어오면 count를 증가시키지 않는다.
    - 다른 코드이고, 이미 DAILY_FREE_LIMIT 만큼 썼다면 429를 던진다.
    - Supabase 클라이언트/쿼리 오류가 나면
      _check_daily_limit_memory() 로 안전하게 fallback 한다.
    """
    today = date.today()

    # 1) Supabase 클라이언트 확보
    try:
        db = get_supabase_client()
    except Exception as e:
        print("[USAGE] get_supabase_client() failed, fallback to memory:", e)
        return _check_daily_limit_memory(user_id, code_hash)

    # 2) 오늘 날짜 row 조회
    try:
        resp = (
            db.table("diagram_usage")
              .select("count,last_code_hash,usage_date")
              .eq("user_id", user_id)
              .eq("usage_date", today.isoformat())
              .limit(1)
              .execute()
        )
    except Exception as e:
        print("[USAGE] select from diagram_usage failed, fallback to memory:", e)
        return _check_daily_limit_memory(user_id, code_hash)

    rows = getattr(resp, "data", None) or []
    if rows:
        row = rows[0]
        count = row.get("count") or 0
        last_hash = row.get("last_code_hash")
        has_row = True
    else:
        count = 0
        last_hash = None
        has_row = False

    is_new_code = (last_hash != code_hash)

    # 3) 새로운 코드 + 이미 한도 초과 ⇒ 429
    if is_new_code and count >= DAILY_FREE_LIMIT:
        print(
            f"[USAGE-DB] LIMIT_EXCEEDED user_id={user_id} "
            f"date={today} count={count}"
        )
        raise HTTPException(
            status_code=429,
            detail={
                "code": "DAILY_LIMIT_EXCEEDED",
                "usage_count": count,
                "daily_free_limit": DAILY_FREE_LIMIT,
            },
        )

    # 4) 같은 코드라면 DB 업데이트 없이 그대로 리턴
    if not is_new_code:
        print(
            f"[USAGE-DB] OK (same code) user_id={user_id} "
            f"date={today} count={count}"
        )
        return count

    # 5) 새로운 코드인 경우: count + 1 후 insert/update
    new_count = count + 1

    try:
        if has_row:
            # 기존 row 업데이트
            db.table("diagram_usage").update(
                {
                    "count": new_count,
                    "last_code_hash": code_hash,
                }
            ).eq("user_id", user_id).eq("usage_date", today.isoformat()).execute()
        else:
            # 신규 row 삽입
            db.table("diagram_usage").insert(
                {
                    "user_id": user_id,
                    "usage_date": today.isoformat(),
                    "count": new_count,
                    "last_code_hash": code_hash,
                }
            ).execute()
    except Exception as e:
        # 사용 자체는 성공시킨 뒤, 로그만 남김
        print("[USAGE-DB] upsert failed, but allow usage:", e)

    print(
        f"[USAGE-DB] OK (new code) user_id={user_id} "
        f"date={today} count={new_count}"
    )
    return new_count



def generate_mermaid_auto(source_code: str, branch_shape: str = "rounded"):
    """
    1) 코드에서 함수 목록을 전부 찾는다.
    2) main이 있으면 main, 없으면 첫 번째 함수를 우선 시도한다.
    3) 선택한 함수에서 본문 추출이 실패하면, 나머지 함수들까지 순차적으로 시도.
    4) 결국 아무 함수도 본문 추출이 안 되면, 어떤 함수들을 발견했는지까지 에러 메시지에 포함.
    """
    # 1) 함수 목록 탐색
    func_list = extract_function_names(source_code)
    # print("[DEBUG] detected functions:", func_list)  # 디버그용

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


@app.get("/debug/supabase")
async def debug_supabase():
    # supabase가 globals에 있는지, 타입이 뭔지 확인
    exists = "supabase" in globals()
    value = globals().get("supabase", None)
    return {
        "exists_in_globals": exists,
        "value_type": str(type(value)),
        "is_none": (value is None),
    }

@app.post("/debug/auth")
async def debug_auth(access_token: str = Form(...)):
    """
    access_token을 보내서, 백엔드에서 어떻게 decode 되는지 확인용.
    (나중에 삭제해도 됨)
    """
    info = verify_access_token(access_token)
    return {
        "ok": True,
        "decoded": info,
    }


def get_supabase_client() -> Client:
    """
    supabase 전역 클라이언트를 안정적으로 가져오는 함수.
    supabase가 None이면 환경변수를 다시 읽어 재생성한다.
    """
    global supabase

    # 이미 전역 supabase가 초기화되어 있다면 그대로 반환
    if supabase is not None:
        return supabase

    # None이면 환경변수로 재생성 시도
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY가 설정되지 않았습니다.")

    supabase = create_client(url, key)
    return supabase


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
    func_name_style: str = Form("short"),
    access_token: str = Form(None),
    user_id: str | None = Form(None),
    user_email: str | None = Form(None),
):
    # 1) 토큰 검증 + 토큰에서 user 정보 꺼내기
    token_info = verify_access_token(access_token)
    token_user_id = token_info.get("user_id")
    token_email   = token_info.get("email")

    # 2) 폼으로 넘어온 값이 있으면 우선, 없으면 토큰 값 사용
    user_id = user_id or token_user_id
    user_email = user_email or token_email

    print(f"[REQ] /api/convert_text user_id={user_id} email={user_email}")

    if not user_id:
        raise HTTPException(status_code=400, detail="MISSING_USER_ID")

    # 같은 코드면 사용 횟수를 올리지 않기 위해 해시를 만든다
    code_hash = make_code_hash(source_code)

    # ===========================
    # 플랜 / 사용량 계산
    # ===========================
    usage_count: int | None = None

    # 1) 무제한 테스트 계정
    is_test_account = (user_email == "exitgiveme@gmail.com")

    # 기본값: Free
    plan_tier   = "free"            # "free" | "pro" | "expert" | "unlimited"
    plan_name   = "Free tier"
    node_limit  = FREE_NODE_LIMIT   # 20
    is_pro_user = False
    subscription_row = None

    if is_test_account:
        # 완전 무제한
        plan_tier   = "unlimited"
        plan_name   = "Test account (unlimited)"
        node_limit  = None
        is_pro_user = True   # 내부적으로는 유료 기능처럼 취급해도 됨
        print("[API] test account: unlimited usage / no node limit")

    else:
        # Supabase에서 active 구독 조회
        subscription_row = get_user_subscription(user_id)

        if subscription_row:
            db_plan_tier = (subscription_row.get("plan_tier") or "").lower()

            if db_plan_tier in ("pro", "expert", "unlimited", "free"):
                plan_tier = db_plan_tier
            else:
                plan_tier = "pro"  # 이상한 값이면 pro 로 폴백

            plan_name = subscription_row.get("plan_name") or plan_tier.title()

            if plan_tier == "expert":
                node_limit = EXPERT_NODE_LIMIT       # 1000
            elif plan_tier == "pro":
                node_limit = PRO_NODE_LIMIT          # 200
            elif plan_tier == "unlimited":
                node_limit = None
            else:
                node_limit = FREE_NODE_LIMIT         # 20

            is_pro_user = plan_tier in ("pro", "expert", "unlimited")

            print(f"[API] active subscription for {user_id}: tier={plan_tier}, row={subscription_row}")
        else:
            # active 구독 없음 → free
            plan_tier   = "free"
            plan_name   = "Free tier"
            node_limit  = FREE_NODE_LIMIT
            is_pro_user = False
            print(f"[API] no subscription for {user_id}, using free tier")

        # Free 플랜에만 일일 사용량 제한 적용
        if plan_tier == "free":
            usage_count = check_daily_limit(user_id, code_hash)

    # ===========================
    # 플로우차트 생성
    # ===========================
    try:
        mermaid, func_name, node_lines, full_signature = generate_mermaid_auto(
            source_code,
            branch_shape=branch_shape,
        )

        # ----- 함수 이름 표시 스타일 적용 (Short / Full) -----
        style = (func_name_style or "short").lower()
        if style not in ("short", "full"):
            style = "short"

        display_short = (func_name or "").strip()
        display_full  = (full_signature or "").strip()

        if style == "full" and display_full:
            display_name = display_full
        else:
            display_name = display_short

        if display_name and func_name:
            pattern_start = r"start\s+" + re.escape(func_name) + r"\s*\(\)?"
            pattern_end   = r"end\s+"   + re.escape(func_name) + r"\s*\(\)?"
            mermaid = re.sub(pattern_start, f"start {display_name}", mermaid)
            mermaid = re.sub(pattern_end,   f"end {display_name}",   mermaid)

        node_count = len(node_lines)

        # ===========================
        # 노드 제한 체크
        # ===========================
        if node_limit is not None and node_count > node_limit:
            return JSONResponse(
                status_code=400,
                content={
                    "mermaid": "",
                    "func_name": func_name,
                    "error": "TOO_MANY_NODES",
                    "error_code": "TOO_MANY_NODES",
                    "node_count": node_count,
                    "node_limit": node_limit,
                    "plan_tier": plan_tier,
                    "plan_name": plan_name,
                    "usage_count": usage_count,
                    "daily_free_limit": DAILY_FREE_LIMIT,
                    "free_node_limit": FREE_NODE_LIMIT,
                    "is_test_account": is_test_account,
                    "is_pro_user": is_pro_user,
                },
            )

        # ===========================
        # 정상 응답
        # ===========================
        return JSONResponse(
            {
                "mermaid": mermaid,
                "func_name": func_name,
                "full_signature": full_signature,
                "node_lines": node_lines,
                "node_count": node_count,
                "usage_count": usage_count,
                "daily_free_limit": DAILY_FREE_LIMIT,
                "free_node_limit": FREE_NODE_LIMIT,
                "is_pro_user": is_pro_user,
                "plan_name": plan_name,
                "plan_tier": plan_tier,
                "node_limit": node_limit,
                "is_test_account": is_test_account,
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

@app.get("/usage")
async def get_usage(user_id: str):
    """
    Return today's usage count for free-tier users.
    """
    # 1) 사용자 플랜 조회
    plan = get_user_plan(user_id)
    plan_tier = plan.get("plan_tier", "free")

    # 무료가 아니면 사용량 체크할 필요 없음
    if plan_tier != "free":
        return {
            "usage_count": None,   # 무제한
            "daily_limit": None    # 제한 없음
        }

    # 2) 오늘 날짜 기준 usage_count 조회
    today = date.today()

    row = supabase.table("diagram_usage") \
        .select("count") \
        .eq("user_id", user_id) \
        .eq("usage_date", today.isoformat()) \
        .single() \
        .execute()

    if row.data:
        usage_count = row.data["count"]
    else:
        usage_count = 0

    return {
        "usage_count": usage_count,
        "daily_limit": DAILY_FREE_LIMIT
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
