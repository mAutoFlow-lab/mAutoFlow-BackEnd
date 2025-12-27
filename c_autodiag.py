#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c_autodiag.py — C 소스에서 Mermaid 다이어그램 생성

기능
- flowchart : 함수 단위 플로우차트 (텍스트 기반 간이 파서)
- callgraph : 정적 호출 그래프 (libclang 사용, 선택 사항)
- sequence  : 런타임 호출 로그(txt)로 시퀀스 다이어그램

특징
- 인자 없이 실행해도 동작: 파일/로그 선택 → 모드 선택 → (callgraph일 때만) 프리셋/상위 폴더 지정
- flowchart 는 libclang, 특수 전처리 없이 동작
- callgraph 는 libclang + 전처리(asm 등 특수 확장 제거) 사용
"""

import os
import sys
import argparse
from collections import defaultdict, deque
import re
from pathlib import Path
import shutil
import subprocess
import html   # ✅ HTML entity (&lt; &gt; &amp; ...) 복원용

# ---------------------------------------------------------------------------
# libclang (callgraph 에서만 사용, 필요할 때만 로드)
# ---------------------------------------------------------------------------
_ci = None  # lazy load

def _ensure_libclang():
    """
    libclang.dll 경로를 자동으로 찾아서 clang.cindex에 등록.
    우선순위:
    1) 환경변수 CLANG_LIBRARY_FILE
    2) 환경변수 LIBCLANG_PATH/libclang.dll
    3) 흔한 설치 경로 후보(LLVM 설치 / MSYS2 mingw64)
    """
    try:
        from clang import cindex  # type: ignore
    except Exception as e:
        print(" clang 파이썬 바인딩을 불러오지 못했습니다. `pip install clang` 필요.", file=sys.stderr)
        raise

    candidates = []
    env_file = os.environ.get("CLANG_LIBRARY_FILE")
    if env_file:
        candidates.append(env_file)

    env_dir = os.environ.get("LIBCLANG_PATH")
    if env_dir:
        candidates.append(os.path.join(env_dir, "libclang.dll"))

    candidates += [
        r"C:\Program Files\LLVM\bin\libclang.dll",
        r"C:\Program Files (x86)\LLVM\bin\libclang.dll",
        r"C:\msys64\mingw64\bin\libclang.dll",
        r"C:\msys64\ucrt64\bin\libclang.dll",
    ]

    tried = []
    for p in dict.fromkeys(candidates):
        if not p:
            continue
        tried.append(p)
        if os.path.isfile(p):
            try:
                cindex.Config.set_library_file(p)
                _ = cindex.Index.create()
                return cindex
            except Exception:
                pass

    # 마지막 시도: 경로 미설정으로 두고 로드
    try:
        _ = cindex.Index.create()
        return cindex
    except Exception:
        msg = "libclang.dll 경로를 찾지 못했습니다.\n시도한 경로:\n  - " + "\n  - ".join(tried) + \
              "\n환경변수 CLANG_LIBRARY_FILE=정확한_dll경로 를 지정하거나, 코드 상단 후보 경로를 수정하세요."
        raise RuntimeError(msg)

# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in s)

def node_spelling(node):
    return node.spelling or node.displayname or str(node.kind).split(".")[-1]

def is_in_mainfile(node):
    try:
        return node.location and node.location.file and node.translation_unit and node.location.file == node.translation_unit.spelling
    except:
        return True

def uniq(seq):
    return list(dict.fromkeys(seq))

# ---------------------------------------------------------------------------
# include 수집 (callgraph 에서만 사용)
# ---------------------------------------------------------------------------

IGNORE_DIR_NAMES = {
    ".git", ".svn", ".hg", "build", "out", "dist",
    "__pycache__", ".vs", ".vscode", "Debug", "Release"
}

def collect_include_dirs(root_dir):
    """root_dir 이하 모든 디렉터리를 -I로 등록 (일부 빌드/숨김 폴더는 제외)"""
    include_args = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIR_NAMES]
        include_args.append(f"-I{dirpath}")
    return uniq(include_args)

# ---------------------------------------------------------------------------
# 프리셋 옵션 (callgraph 전용)
# ---------------------------------------------------------------------------

def preset_msvc():
    from pathlib import Path as _Path

    llvm_candidates = [
        r"C:\Program Files\LLVM\lib\clang\18\include",
        r"C:\Program Files\LLVM\lib\clang\17\include",
        r"C:\Program Files (x86)\LLVM\lib\clang\18\include",
        r"C:\Program Files (x86)\LLVM\lib\clang\17\include",
    ]
    llvm_inc = [f'-I"{p}"' for p in llvm_candidates if os.path.isdir(p)]

    vs_roots = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Tools\MSVC",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC",
    ]
    vs_includes = []
    for root in vs_roots:
        if not os.path.isdir(root):
            continue
        versions = sorted(os.listdir(root), reverse=True)
        for ver in versions:
            inc = os.path.join(root, ver, "include")
            if os.path.isdir(inc):
                vs_includes.append(f'-I"{inc}"')
                break
        if vs_includes:
            break

    sdk_root = r"C:\Program Files (x86)\Windows Kits\10\Include"
    sdk_includes = []
    if os.path.isdir(sdk_root):
        versions = sorted(os.listdir(sdk_root), reverse=True)
        for ver in versions:
            base = os.path.join(sdk_root, ver)
            for sub in ("ucrt", "um", "shared", "cppwinrt"):
                d = os.path.join(base, sub)
                if os.path.isdir(d):
                    sdk_includes.append(f'-I"{d}"')
            if sdk_includes:
                break

    return [
        "-x", "c",
        "-std=c11",
        "-fms-compatibility",
        "-fms-extensions",
    ] + llvm_inc + vs_includes + sdk_includes

def preset_mingw():
    mingw_candidates = [
        r"C:\msys64\mingw64\include",
        r"C:\msys64\ucrt64\include",
    ]
    llvm_candidates = [
        r"C:\Program Files\LLVM\lib\clang\18\include",
        r"C:\Program Files\LLVM\lib\clang\17\include",
    ]
    incs = [f'-I"{p}"' for p in mingw_candidates + llvm_candidates if os.path.isdir(p)]
    target = []
    if os.path.isdir(r"C:\msys64\mingw64"):
        target = ["-target", "x86_64-w64-windows-gnu"]
    elif os.path.isdir(r"C:\msys64\ucrt64"):
        target = ["-target", "x86_64-w64-windows-gnu"]

    return [
        "-x", "c",
        "-std=c11",
    ] + target + incs

def preset_none():
    return ["-x", "c", "-std=c11"]

# ---------------------------------------------------------------------------
# 전처리기 (asm / 특수 키워드 제거 — callgraph 전용)
# 이름에서 RL78 용어 제거
# ---------------------------------------------------------------------------

# @주소 제거
_ADDR_ATTR = re.compile(r'@\s*0x[0-9A-Fa-f]+(\.[0-9]+)?')

# asm("...") / __asm("...")
_ASM_CALL = re.compile(r'\b__?asm\s*\(\s*".*?"\s*\)\s*;?', re.S)

# asm { ... } / __asm { ... }
_ASM_BLOCK = re.compile(r'\b__?asm\s*\{.*?\}', re.S)

# #pragma asm ... #pragma endasm
_PRAGMA_ASM_BLOCK = re.compile(r'#\s*pragma\s+asm.*?#\s*pragma\s+endasm', re.S | re.I)

# __attribute__((...)), __declspec(...)
_ATTR = re.compile(r'__attribute__\s*\(\s*\(.*?\)\s*\)', re.S)
_DECLSPEC = re.compile(r'__declspec\s*\(.*?\)', re.S)
_ASM_LINE = re.compile(r'\b__?asm\b.*')

# 벤더 전용 키워드(일반 C 컴파일러가 모를 수 있는 것들)
_VENDOR_KEYWORDS = [
    "__sfr", "__bit", "__near", "__far", "__saddr", "__tiny",
    "__no_init", "__root", "__interrupt", "__evenaccess",
    "__data", "__at", "__IO", "__I", "__O",
    "sfr", "sfrb", "sfrw", "sfrp", "bit",
]

def preprocess_vendor_code(code: str) -> str:
    """임베디드 컴파일러 전용 확장 문법/asm 등을 최대한 제거."""
    code = _ADDR_ATTR.sub("", code)
    code = _PRAGMA_ASM_BLOCK.sub("", code)
    code = _ASM_BLOCK.sub("", code)
    code = _ASM_CALL.sub("", code)
    code = _ASM_LINE.sub("", code)

    for kw in _VENDOR_KEYWORDS:
        code = re.sub(rf"\b{kw}\b", "", code)

    code = _ATTR.sub("", code)
    code = _DECLSPEC.sub("", code)

    # 공백/탭만 정리, 줄바꿈은 유지
    code = re.sub(r"[ \t]+", " ", code)
    return code

def preprocess_tree_vendor(src_root: str, out_root: str) -> None:
    """src_root 전체를 클린한 트리(out_root)로 미러링."""
    src_root = Path(src_root).resolve()
    out_root = Path(out_root).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    exts = {".c", ".h", ".hpp", ".hh", ".inl"}

    count = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        dirnames[:] = [d for d in dirnames if d not in {
            ".git", ".svn", "__pycache__", "build", "out", "Debug", "Release"
        }]
        rel = Path(dirpath).relative_to(src_root)
        dst = out_root / rel
        dst.mkdir(parents=True, exist_ok=True)

        for f in filenames:
            src_file = Path(dirpath) / f
            dst_file = dst / f
            if src_file.suffix.lower() in exts:
                txt = src_file.read_text(encoding="utf-8", errors="ignore")
                dst_file.write_text(preprocess_vendor_code(txt), encoding="utf-8")
                count += 1
            else:
                shutil.copy2(src_file, dst_file)

    print(f"전처리(asm 등 특수 확장 제거): {count}개 파일 정리됨")

# ---------------------------------------------------------------------------
# (New) 미니 파서 + 분기 지원 플로우차트  — flowchart 전용, libclang 미사용
# ---------------------------------------------------------------------------
def sanitize_func_name(s: str) -> str:
    """
    UI/외부에서 'FUNC(...) Foo(...' 같은 문자열이 넘어와도
    마지막 식별자(Foo)만 뽑아서 함수명으로 사용.
    """
    s = (s or "").strip()
    if not s:
        return s
    # '(' 이전까지만 보되, 마지막 identifier를 뽑는다
    head = s.split("(", 1)[0].strip()
    m = re.search(r"([A-Za-z_]\w*)\s*$", head)
    return m.group(1) if m else s


def remove_comments(code: str) -> str:
    """//, /* */ 주석 제거 (대충이지만 웬만하면 잘 동작)"""
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)   # 블록 주석
    code = re.sub(r"//.*?$", "", code, flags=re.M)      # 라인 주석
    return code

def extract_function_body(code: str, func_name: str, macros: dict | None = None) -> str:
    """
    텍스트에서 원하는 함수 하나의 본문만 뽑아냄.
    예: void main(void) { ... } 에서 { ... } 부분.
    """
    code_nc = remove_comments(code)

    # f-string / format 안 쓰고, 문자열을 이어 붙여서 만든다.
    pattern = re.compile(
        r'^\s*'
        r'(?:(?:static|STATIC|inline|extern|register)\s+)*'  # ✅ [NEW] FUNC 앞 수식어 허용
        r'(?:FUNC\s*\((?:[^()]|\([^()]*\))*\)\s*)?'   # ✅ 중첩 괄호 허용
        r'(?:[A-Za-z_][\w\s\*]*\s+)?'                 # 반환형/수식어 (옵션)
        + re.escape(func_name) +
        r'\s*\('
        r'(?:[^()]|\([^()]*\))*'
        r'\)\s*(?:\{|\n\s*\{)',
        re.MULTILINE,
    )

    m = pattern.search(code_nc)
    if not m:
        raise ValueError(f"Function {func_name} not found in code.")

    start_brace = m.end() - 1  # '{' 위치

    # ----------------------------
    # (A) 1차: 기존 방식 그대로 시도
    # ----------------------------
    def _scan_to_matching_brace(text: str, start_idx: int) -> int | None:
        depth = 1
        i = start_idx + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        return (i - 1) if depth == 0 else None

    end_brace = _scan_to_matching_brace(code_nc, start_brace)
    if end_brace is not None:
        return code_nc[start_brace + 1 : end_brace]

    # ----------------------------
    # (B) 2차: 전처리(#if/#else/#endif) 적용 후 다시 매칭
    # ----------------------------
    macros = macros or {}

    tail = code_nc[start_brace:]              # '{'부터 끝까지
    tail_lines = tail.splitlines()
    tail_lines = splice_backslash_lines(tail_lines)
    tail_lines = mini_preprocess_lines(tail_lines, macros)

    tail_pp = "\n".join(tail_lines)
    # 전처리 후에도 첫 글자가 '{'가 아닐 수 있으니 안전하게 찾기
    first_brace = tail_pp.find("{")
    if first_brace == -1:
        raise ValueError(f"Function {func_name} has mismatched curly brackets.")

    end2 = _scan_to_matching_brace(tail_pp, first_brace)
    if end2 is None:
        raise ValueError(f"Function {func_name} has mismatched curly brackets.")

    return tail_pp[first_brace + 1 : end2]


def extract_function_names(code: str):
    """
    매우 단순한 함수 이름 추출기.
    - void _EntryPoint(void) { ... }
    - static void Foo(void)
    - uint8_t Bar(int x, int y)
    - AUTOSAR 스타일: FUNC(type, memclass) Foo(
    """
    code_nc = remove_comments(code)

    func_pattern = re.compile(
        r"""
        ^\s*
        (?:(?:static|STATIC|inline|extern|register)\s+)*      # ✅ [NEW] FUNC 앞 수식어 허용
        (?:FUNC\s*\((?:[^()]|\([^()]*\))*\)\s*)?       # ✅ 중첩 괄호 허용
        (?:[A-Za-z_][\w\s\*]*\s+)?                     # 일반 반환형/수식어 (옵션)
        ([A-Za-z_]\w*)                                 # 함수 이름
        \s*\(
            (?:[^()]|\([^()]*\))*                      # 파라미터
        \)
        \s*(?:\{|\n\s*\{)
        """,
        re.MULTILINE | re.VERBOSE,
    )
    
    names = list(dict.fromkeys(func_pattern.findall(code_nc)))

    # 제어문/예약어 + 대표적인 AUTOSAR 매크로는 함수에서 제외
    blacklist = {
        "if", "else", "switch", "case", "default",
        "for", "while", "do",
        "return", "sizeof",
        "struct", "union", "enum",
        "static", "goto", "break", "continue",

        "FUNC", "P2VAR", "P2CONST", "P2FUNC",
    }

    return [n for n in names if n not in blacklist]


def _block_is_effectively_empty(lines, start, end):
    """
    start ~ end-1 구간이 실질적으로 비어 있는지 검사.
    - 공백, { }, ;, 한 줄 주석, /* ... */ 형식만 있는 줄은 모두 무시.
    """
    for i in range(start, end):
        s = lines[i].strip()
        if not s:
            continue

        # 여는/닫는 중괄호나 세미콜론만 있는 경우
        if s in ("{", "}", ";"):
            continue

        # // 로 시작하는 한 줄 주석
        if s.startswith("//"):
            continue

        # 한 줄짜리 /* ... */ 주석
        if s.startswith("/*") and s.endswith("*/"):
            continue

        # 위 경우가 모두 아니면 "실질적인 코드가 있다"로 판단
        return False

    # 끝까지 돌았는데 유의미한 코드가 없으면 빈 블록
    return True

_pre_if_pattern = re.compile(r'^\s*#(if|ifdef|ifndef|elif|else|endif)\b(.*)$')


# C 정수 리터럴 suffix 제거 + (필요 시) 8진수 변환
_C_INT_LIT = re.compile(
    r"""
    (?P<num>
        0[xX][0-9A-Fa-f]+ |      # hex
        0[bB][01]+        |      # binary (gcc/ext 포함)
        0[0-7]+           |      # octal (C 스타일)
        [0-9]+                   # decimal
    )
    (?P<suf>[uUlL]+)?            # U, L, UL, LL, ULL 등
    \b
    """,
    re.VERBOSE,
)

def _normalize_c_int_literals(expr: str) -> str:
    """
    expr 안의 C 정수 리터럴에서 suffix(U/L/UL/LL...) 제거.
    또한 C의 8진수(예: 0755)를 Python eval 가능하도록 0o755로 변환.
    """
    def repl(m: re.Match) -> str:
        num = m.group("num")
        # octal: 0[0-7]+ 이고 hex/binary가 아닌 경우만 변환
        if len(num) >= 2 and num[0] == "0" and num[1].isdigit() and not (
            num.startswith(("0x", "0X", "0b", "0B"))
        ):
            return "0o" + num[1:]
        return num

    return _C_INT_LIT.sub(repl, expr)



def _eval_pp_condition(expr: str, macros: dict) -> bool:
    """
    #if / #elif 조건을 단순 평가하는 미니 버전.
    지원:
      - defined(FLAG), !defined(FLAG)
      - FLAG, !FLAG
      - 숫자 상수 (0, 1, 2 ...)
      - TEST > 1 같이 간단한 비교식 (매크로 값을 숫자로 치환 후 eval)
    """
    expr = expr.strip()
    expr = _normalize_c_int_literals(expr)
    
    if not expr:
        return False

    # defined(FLAG) 처리
    def repl_defined(m):
        name = m.group(1)
        return "1" if name in macros else "0"

    expr = re.sub(r'defined\s*\(\s*([A-Za-z_]\w*)\s*\)', repl_defined, expr)

    # 식 안에 남은 식별자(매크로 이름)들을 숫자로 치환
    def repl_ident(m):
        name = m.group(0)
        val = macros.get(name, None)
        if val is None:
            return "0"
        if isinstance(val, bool):
            return "1" if val else "0"

        s = str(val).strip()
        parsed = _try_parse_c_int_literal(s)
        if parsed is not None:
            return str(parsed)

        # 숫자가 아니면 그냥 1로
        return "1"

    expr = re.sub(r'\b([A-Za-z_]\w*)\b', repl_ident, expr)

    # C 스타일 논리 연산자 → Python으로 변환
    expr = expr.replace("&&", " and ").replace("||", " or ")
    # 단항 ! 를 not 으로 ( != 는 그대로 두기 위해 주의)
    expr = re.sub(r'(?<![=!<>])!(?!=)', ' not ', expr)

    # 아주 간단한 안전장치: 허용 문자만 남았는지 체크
    if re.search(r'[^0-9\s\(\)\+\-\*/%<>=!andornot]', expr):
        # 알 수 없는 문자가 끼어 있으면 False 로 처리
        return False

    try:
        return bool(eval(expr, {"__builtins__": {}}, {}))
    except Exception:
        return False

def mini_preprocess_lines(lines: list[str], macros: dict) -> list[str]:
    """
    간단 전처리기:
      - #if / #ifdef / #ifndef / #elif / #else / #endif 만 처리
      - 살아있는 코드 줄만 남기고, 전처리 지시문 라인은 모두 제거
    """
    out: list[str] = []
    stack: list[dict] = []
    active = True  # 현재 줄이 활성 상태인지

    for line in lines:
        m = _pre_if_pattern.match(line)
        if not m:
            # 전처리 지시문이 아니면, active 일 때만 출력
            if active:
                out.append(line)
            continue

        kind = m.group(1)
        rest = m.group(2).strip()

        if kind == "if":
            parent_active = active
            cond_true = _eval_pp_condition(rest, macros) if parent_active else False
            frame = {
                "parent_active": parent_active,
                "any_taken": bool(cond_true),
                "current_active": bool(cond_true),
            }
            stack.append(frame)
            active = frame["current_active"]
            continue

        if kind == "ifdef":
            parent_active = active
            cond_true = (rest in macros) if parent_active else False
            frame = {
                "parent_active": parent_active,
                "any_taken": bool(cond_true),
                "current_active": bool(cond_true),
            }
            stack.append(frame)
            active = frame["current_active"]
            continue

        if kind == "ifndef":
            parent_active = active
            cond_true = (rest not in macros) if parent_active else False
            frame = {
                "parent_active": parent_active,
                "any_taken": bool(cond_true),
                "current_active": bool(cond_true),
            }
            stack.append(frame)
            active = frame["current_active"]
            continue

        if kind == "elif":
            if not stack:
                # 이상한 구조면 무시
                continue
            frame = stack[-1]
            parent_active = frame["parent_active"]
            if not parent_active:
                frame["current_active"] = False
                active = False
                continue
            if frame["any_taken"]:
                frame["current_active"] = False
                active = False
                continue
            cond_true = _eval_pp_condition(rest, macros)
            frame["current_active"] = bool(cond_true)
            if cond_true:
                frame["any_taken"] = True
            active = frame["current_active"]
            continue

        if kind == "else":
            if not stack:
                continue
            frame = stack[-1]
            parent_active = frame["parent_active"]
            if not parent_active:
                frame["current_active"] = False
                active = False
                continue
            if frame["any_taken"]:
                frame["current_active"] = False
                active = False
            else:
                frame["current_active"] = True
                frame["any_taken"] = True
                active = True
            continue

        if kind == "endif":
            if not stack:
                continue
            frame = stack.pop()
            active = frame["parent_active"]
            continue

    return out


# C 정수 리터럴 (10진/16진/2진/8진) + 접미사(U/L/UL/ULL...) 처리
_INT_LIT_RE = re.compile(
    r"""^\s*
    (?P<num>
        0[xX][0-9A-Fa-f]+ |
        0[bB][01]+ |
        0[0-7]* |
        [1-9][0-9]* |
        0
    )
    (?P<suf>[uUlL]{0,4})
    \s*$""",
    re.X,
)

def _try_parse_c_int_literal(s: str) -> int | None:
    if s is None:
        return None
    m = _INT_LIT_RE.match(str(s))
    if not m:
        return None
    num = m.group("num")

    try:
        # 0x / 0b 는 int(...,0)로 OK
        if num.startswith(("0x", "0X", "0b", "0B")):
            return int(num, 0)

        # "077" 같은 C 스타일 8진수 처리
        if len(num) >= 2 and num[0] == "0" and num[1].isdigit():
            return int(num, 8)

        # 나머지 (0, 10진)
        return int(num, 10)
    except Exception:
        return None

def parse_macro_string(s: str) -> dict:
    """
    입력 예:
      "DEBUG;TEST=2;RELEASE"
    출력:
      {"DEBUG": 1, "TEST": 2, "RELEASE": 1}
    """
    macros = {}
    s = (s or "").strip()
    if not s:
        return macros

    parts = [p.strip() for p in re.split(r"[;,\n]+", s) if p.strip()]
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            # 숫자면 int, 아니면 문자열/1 처리
            parsed = _try_parse_c_int_literal(v)
            if parsed is not None:
                macros[k] = parsed
            else:
                macros[k] = v
        else:
            macros[p] = 1
    return macros


def splice_backslash_lines(lines: list[str]) -> list[str]:
    """
    C preprocessor line-splicing:
    라인이 '\' 로 끝나면 다음 라인과 이어붙여서 하나의 논리 라인으로 만든다.
    (트레일링 공백 뒤의 '\' 도 처리)
    """
    out = []
    buf = ""
    for line in lines:
        # 줄 끝 공백 제거 후 '\' 체크
        stripped_r = line.rstrip("\r\n")
        if stripped_r.rstrip().endswith("\\"):
            # '\' 제거하고 이어붙이기(한 칸 띄워서)
            part = stripped_r.rstrip()
            part = part[:-1].rstrip()  # remove trailing '\'
            buf += (part + " ")
        else:
            if buf:
                out.append(buf + stripped_r)
                buf = ""
            else:
                out.append(stripped_r)
    if buf:
        out.append(buf.rstrip())
    return out



class StructuredFlowEmitter:
    """
    텍스트 기반 단순 제어 흐름 파서.
    - if / else / else if  : True / False 분기 + merge
      * else 블록이 비어있으면 else 박스 없이 False → merge 로 바로 연결
    - while                : True(루프 안) / False(탈출) + 루프백
    - for(init; cond; post):
        init   → 사각형
        cond   → 다이아몬드(조건만)
        post   → 바디 뒤 사각형 → cond 로 루프백
      * for(;;) : 무한루프 → False 분기/True 라벨 없이 바디 ↔ 조건만 반복
    - switch / case        : case별 가지치기 + merge (case 안 내용은 요약)
    - 그 외 라인            : 순차 action 노드
    """

    def _extract_if_condition(self, header_text: str) -> str:
        """
        'if (cond) stmt;' 형태에서 'if (cond)' 부분만 잘라낸다.
        멀티라인 조건도 괄호 깊이 계산해서 닫히는 ')' 까지만 사용.
        """
        m = re.search(r"\bif\s*\(", header_text)
        if not m:
            return header_text
        i = m.end()
        depth = 1
        while i < len(header_text) and depth > 0:
            ch = header_text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            return header_text[:i]  # 'if ( ... )' 까지만
        return header_text

    def __init__(self, func_name: str, branch_shape: str = "rounded", macros: dict | None = None):
        self.func_name = sanitize_func_name(func_name)
        self.branch_shape = branch_shape
        self.lines = []
        self.node_id = 0
        self.infinite_loop_nodes = set()
        self.end_node = None

        # 미니 전처리기용 매크로 정의 (예: {"DEBUG": "1", "TEST": "2"})
        self.macros = macros or {}

        # 루프 안의 break / continue 노드를 관리하기 위한 스택
        self.break_stack = []      # [ [Nid, Nid, ...], ... ]
        self.continue_stack = []   # [ [Nid, Nid, ...], ... ]

        # switch depth 깊이
        self.switch_depth = 0

        # [NEW] label / goto 처리용
        self.label_nodes = {}      # label_name -> node_id
        self.goto_pending = []     # (goto_node_id, label_name)

        self.goto_nodes = set()    # goto 문 노드들

        # 노드 ID -> 함수 본문 내 라인 인덱스(0-based, body.splitlines() 기준)
        self.node_line_map = {}

    def _is_loop_control_node(self, nid: str) -> bool:
        """현재 가장 안쪽 루프에서 break/continue 로 쓰이는 노드인지 확인"""
        if self.break_stack and nid in self.break_stack[-1]:
            return True
        if self.continue_stack and nid in self.continue_stack[-1]:
            return True
        return False  

    # [NEW] 최초 진입 노드 기록용 헬퍼
    def _register_entry(self, entry_holder, nid: str) -> None:
        """
        entry_holder 가 리스트([])일 때, 그 시퀀스에서
        가장 처음 생성된 노드 ID를 한 번만 기록.
        """
        if entry_holder is not None and not entry_holder:
            entry_holder.append(nid)

    def _bind_node_line(self, nid: str, line_idx: int) -> None:
        """
        node id 를 함수 본문 내 라인 인덱스에 매핑.
        line_idx 는 body.splitlines() 기준 0-based.
        """
        if line_idx is None:
            return
        if line_idx < 0:
            return
        self.node_line_map[nid] = int(line_idx)

        
    # ---- 공통 유틸 ----
    def nid(self) -> str:
        self.node_id += 1
        return f"N{self.node_id}"

    def add(self, s: str) -> None:
        self.lines.append(s)

    def _clean_label(self, line: str) -> str:
        s = line.strip()
        s = html.unescape(s)
        s = s.rstrip().rstrip('{}').rstrip()

        # (선택) 라벨 표시 안전문자 치환
        s = (s
             .replace("<<", "＜＜")
             .replace(">>", "＞＞")
             .replace("<=", "≤")
             .replace(">=", "≥")
             .replace("!=", "≠")
             .replace("<", "＜")
             .replace(">", "＞")
             .replace("&", "＆")
             .replace("|", "｜")
        )

        max_len = 220
        if len(s) > max_len:
            s = s[:max_len - 3] + "..."

        return s.replace('"', "'")



    def _clean_cond_label(self, line: str) -> str:
        s = line.strip()
        s = html.unescape(s)  # 혹시 들어온 엔티티는 먼저 복원
        s = s.rstrip().rstrip('{}').rstrip()

        # 공백 정리 (연속 공백 → 한 칸)
        s = " ".join(s.split())

        # ✅ Mermaid가 SVG 텍스트로 렌더링할 때 < > & | 등을 엔티티 문자열로 바꿔버리는 케이스가 있음.
        # 그래서 "표시용 라벨"에서만 안전 문자로 치환해서 그대로 보이게 만든다.
        # (코드 로직에는 영향 없음: 출력 라벨만 바뀜)
        s = (s
             .replace("<<", "＜＜")
             .replace(">>", "＞＞")
             .replace("<=", "≤")
             .replace(">=", "≥")
             .replace("!=", "≠")
             .replace("<", "＜")
             .replace(">", "＞")
        )

        # & / | 는 먼저 단일 문자를 안전 문자로 바꾼 뒤, 논리연산자 줄바꿈 처리
        s = s.replace("&", "＆").replace("|", "｜")

        # 줄바꿈 가독성 (논리 연산자)
        s = s.replace("＆＆", "\n＆＆").replace("｜｜", "\n｜｜")

        return s.replace('"', "'")


    def _make_cond_node(self, node_id: str, label: str) -> str:
        """
        branch_shape 설정에 따라 조건 노드 모양을 바꾼다.
        - "rounded" : ( )  → 둥근 사각형 (stadium)
        - "diamond" : { }  → 마름모
        """
        if self.branch_shape == "diamond":
            # 마름모 (decision)
            return f'{node_id}{{"{label}"}}:::cond'
        else:
            # 둥근 사각형
            return f'{node_id}("{label}"):::cond'   

    def _classify_simple(self, line: str) -> str:
        """단순 라인에 대해서만 terminator / action 구분"""
        s = line.strip().lower()
        if s.startswith("return"):
            return "terminator"
        return "action"

    def _find_block(self, lines, brace_line_idx):
        """
        '{' 가 포함된 라인 인덱스를 받아,
        그 블록의 (내용 시작 인덱스, 내용 끝(exclusive), 블록 뒤 첫 라인 인덱스)를 반환.
        """
        open_count = 0
        i = brace_line_idx
        n = len(lines)
        while i < n:
            text = lines[i]
            for ch in text:
                if ch == "{":
                    open_count += 1
                elif ch == "}":
                    open_count -= 1
            i += 1
            if open_count == 0:
                break

        closing_line = i - 1
        start = brace_line_idx + 1
        end_exclusive = closing_line + 1
        after_idx = i
        return start, end_exclusive, after_idx

    # ---- 메인 엔트리 ----
    def emit_from_body(self, body: str) -> str:
        """
        함수 본문(중괄호 안) 문자열을 받아 Mermaid flowchart 텍스트 반환.
        """
        self.lines = []
        self.node_id = 0

        raw_lines = body.splitlines()
        raw_lines = splice_backslash_lines(raw_lines)   # ✅ '\'+newline 라인 결합

        # 미니 전처리기: 매크로가 지정되어 있으면, 먼저 전처리 통과
        if self.macros:
            raw_lines = mini_preprocess_lines(raw_lines, self.macros)        

        self.add("flowchart TD")
        self.add("classDef term fill:#eaffea,stroke:#66cc66,stroke-width:1px;")   # start/end 연두색
        self.add("classDef cond fill:#ffe6cc,stroke:#ff9900,stroke-width:1px;")  # 조건문 주황색
        self.add("classDef preprocess fill:#fff8cc,stroke:#e6c200,stroke-width:1px;")  # 전처리기 노랑
        self.add("classDef merge fill:#c0c0c0,stroke:#555555,stroke-width:1px;")       # merge 회색
        self.add("%% style above, nodes below")

        # main() 인지 여부
        is_main = (self.func_name == "main")

        # start 노드 (한 번만 추가, 공백 없이)
        start = self.nid()
        self.add(f'{start}(["start {self.func_name}()"]):::term')

        # main이 아닐 때만 end 노드 추가
        end = None
        if not is_main:
            end = self.nid()
            self.add(f'{end}(["end {self.func_name}()"]):::term')

        # end 노드를 emitter 저장 (return 시 바로 연결하기 위해)
        self.end_node = end    

        # 본문 파싱
        last = self._parse_sequence(
            raw_lines, 0, len(raw_lines), start, first_edge_label=None, is_top_level=True
        )

        # main() 이 아니고, 마지막 노드가 무한루프 노드가 아닐 때만 end 연결
        if (
            end is not None
            and (last not in self.infinite_loop_nodes)
            and (last != end)
            and (last not in self.goto_nodes)     # [NEW]
        ):
            self.add(f"{last} --> {end}")

        # [NEW] 모든 goto → label 연결
        for nid, label in self.goto_pending:
            if not label:
                continue
            target = self.label_nodes.get(label)
            if target:
                self.add(f"{nid} --> {target}")            

        return "\n".join(self.lines)

    # ---- 시퀀스 파서 (재귀) ----
    def _parse_sequence(
        self,
        lines,
        start_idx,
        end_idx,
        prev_node,
        first_edge_label=None,
        entry_holder=None,
        is_top_level=False,
    ):
        """
        [start_idx, end_idx) 구간을 순차적으로 파싱.
        prev_node에서 시작하여 마지막 노드 id를 반환.
        첫 엣지에만 first_edge_label(True/False 등)을 사용할 수 있음.
        """
        i = start_idx
        n = end_idx
        cur_prev = prev_node          # 진행 중인 마지막 노드
        first_label = first_edge_label
        any_node_created = False

        while i < n:
            start_line = i
            raw = lines[i]

            # [NEW] C line-continuation: 끝이 '\' 인 줄은 다음 줄과 같은 "한 줄"로 합친다.
            # - 전처리기(#if/#elif/...) 뿐 아니라, 매크로 호출/주석 라인에 붙는 '\'도 같이 정리됨
            logical = raw.rstrip()
            j = i
            while logical.endswith("\\") and (j + 1) < n:
                logical = logical[:-1].rstrip() + " " + lines[j + 1].lstrip()
                j += 1
            raw = logical
            next_i = j + 1  # i..j까지 소비했으니 다음은 j+1

            # [NEW] 함수 호출/표현식이 괄호 때문에 줄바꿈된 경우도 "한 줄"로 합치기
            # 예) Det_ReportError(a,
            #                   b,
            #                   c);
            stripped0 = raw.strip()
            is_preproc = stripped0.startswith("#")
            is_ctrl = re.match(r"^\s*(if|for|while|switch)\b", raw) is not None

            if (not is_preproc) and (not is_ctrl):
                # 괄호 밸런스가 남아있으면 다음 줄들을 계속 붙인다
                paren_balance = raw.count("(") - raw.count(")")
                saw_paren = ("(" in raw)


                # [NEW] 함수 호출이 다음 줄에서 '(' 로 시작하는 스타일 지원
                # 예)
                #   x = foo
                #       (a, b);
                # → 첫 줄에는 '(' 가 없어서 기존 로직으로는 결합이 시작되지 않음
                if not saw_paren:
                    kk = next_i
                    while kk < n and not lines[kk].strip():
                        kk += 1
                    # 현재 줄이 식별자/함수명으로 끝나고, 다음 유효 줄이 '(' 로 시작하면 결합 시작
                    callee_tail_re = r"""
                    (?x)                                # verbose
                    (?:\(\s*\*\s*)?                     # optional '(*'
                    [A-Za-z_]\w*                        # identifier
                    (?:\s*\))?                          # optional ')'
                    (?:                                 # postfix chain
                        \s*(?:->|\.)\s*[A-Za-z_]\w*     # ->member or .member
                      | \s*\[[^\]]*\]                   # [index]
                      | \s*\(\s*\)                      # postfix call: ()
                    )*\s*$                              # end
                    """

                    if kk < n and lines[kk].lstrip().startswith("(") and re.search(callee_tail_re, raw.rstrip()):
                        raw = raw.rstrip() + " " + lines[kk].strip()
                        # 이제 '(' 를 봤으니 balance 기반 결합을 이어서 수행
                        paren_balance = raw.count("(") - raw.count(")")
                        saw_paren = True
                        next_i = kk + 1

                k = next_i
                # 괄호를 봤고 balance가 0이 될 때까지 이어붙임
                while saw_paren and paren_balance > 0 and k < n:
                    nxt = lines[k].strip()
                    # 빈 줄은 스킵(붙이진 않음)
                    if not nxt:
                        k += 1
                        continue
                    raw = raw.rstrip() + " " + nxt
                    paren_balance += nxt.count("(") - nxt.count(")")
                    k += 1

                # i~(k-1)까지 소비했으면 next_i 갱신
                next_i = k

            # [NEW] 연산자 줄바꿈 결합 (두 스타일 모두 지원)
            # 1) operator-leading:  다음 줄이 '<<', '&&' 등 연산자로 시작
            # 2) operator-trailing: 현재 줄이 '<<', '&&' 등 연산자로 끝나고 다음 줄이 식별자/상수로 시작
            op_leading_re = r"^(<<|>>|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|[+\-*/%&|^=<>?:])"
            op_trailing_re = r"(<<|>>|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|==|!=|<=|>=|[+\-*/%&|^=<>?:])\s*$"

            stripped0 = raw.strip()
            is_preproc = stripped0.startswith("#")
            is_ctrl = re.match(r"^\s*(if|for|while|switch)\b", raw) is not None
            if (not is_preproc) and (not is_ctrl):
                k = next_i
                while k < n and not raw.strip().endswith(";"):
                    nxt_raw = lines[k]
                    nxt = nxt_raw.strip()

                    if not nxt:
                        k += 1
                        continue

                    # 전처리기/블록 경계/라벨/case 는 여기서 끊기
                    if (
                        nxt.startswith("#")
                        or nxt in ("{", "}")
                        or re.match(r"^[A-Za-z_]\w*\s*:\s*$", nxt)
                        or re.match(r"^(case\b|default\s*:)", nxt)
                    ):
                        break

                    # (0) close-only continuation:
                    #     ')', ');', '),', '))', '));', '})', '});', '},', '])', '];', '],', ')),' 등
                    #     "닫는 괄호/브레이스/대괄호 + (선택) 세미콜론/콤마" 만 있는 줄은
                    #     앞 statement에 붙여서 ');' 같은 조각 노드가 생기는 것을 막는다.
                    if re.match(r"^[\)\]\}]+[;,]?$", nxt):
                        raw = raw.rstrip() + " " + nxt
                        k += 1
                        continue

                    # (1) 다음 줄이 연산자로 시작하면 붙이기
                    if re.match(op_leading_re, nxt):
                        raw = raw.rstrip() + " " + nxt
                        k += 1
                        continue

                    # (2) 현재 줄이 연산자로 끝나면, 다음 줄이 식별자/상수로 시작하는 경우도 붙이기
                    if re.search(op_trailing_re, raw.rstrip()):
                        raw = raw.rstrip() + " " + nxt
                        k += 1
                        continue

                    break

                next_i = k

            # [NEW] operator 결합으로 '(' 가 새로 붙은 경우가 있음 (예: "a =\n (uint16)(" )
            #       그래서 paren_balance를 "다시" 계산해서 ')' 닫힐 때까지 한 번 더 합친다.
            stripped0 = raw.strip()
            is_preproc = stripped0.startswith("#")
            is_ctrl = re.match(r"^\s*(if|for|while|switch)\b", raw) is not None
            if (not is_preproc) and (not is_ctrl):
                paren_balance = raw.count("(") - raw.count(")")
                saw_paren = ("(" in raw)
                k = next_i
                while saw_paren and paren_balance > 0 and k < n:
                    nxt = lines[k].strip()
                    if not nxt:
                        k += 1
                        continue
                    raw = raw.rstrip() + " " + nxt
                    paren_balance += nxt.count("(") - nxt.count(")")
                    k += 1
                next_i = k


            stripped = raw.strip()
            if not stripped or stripped in ("{", "}", ";"):
                i = next_i
                continue

            # ---- 주석만 있는 줄은 무시 ----
            # (혹시 remove_comments()가 적용되지 않은 경로에서 사용할 때를 대비)
            if stripped.startswith("//"):
                i = next_i
                continue
            if stripped.startswith("/*") and stripped.endswith("*/"):
                i = next_i
                continue

            # 괄호나 &&, || 만 있는 라인은
            # 긴 조건식을 줄바꿈한 포매팅용이므로 노드로 만들지 않고 무시
            # 예: 
            #   if (cond1
            #       && cond2
            #       )
            if re.fullmatch(r"[()&|]+", stripped):
                i = next_i
                continue


            lstrip = raw.lstrip()

            # ---- 전처리기 (#if / #elif / #else / #endif / #ifdef / #ifndef) ----
            if stripped.startswith(("#if", "#elif", "#else", "#endif", "#ifdef", "#ifndef")):
                nid = self.nid()
                self._bind_node_line(nid, start_line)   # ✅ 하이라이트는 첫 줄 기준
                label = self._clean_label(stripped)
                self.add(f'{nid}["{label}"]:::preprocess')

                if first_label and cur_prev is not None:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                elif cur_prev is not None:
                    self.add(f"{cur_prev} --> {nid}")
                self._register_entry(entry_holder, nid)
                
                cur_prev = nid
                any_node_created = True
                i = next_i
                continue

            # ---- label:  (예: NEGATIVE:) ----
            m_label = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*$", raw)
            if m_label:
                nid = self.nid()
                self._bind_node_line(nid, i)
                label_txt = self._clean_label(raw)
                self.add(f'{nid}["{label_txt}"]')

                if first_label and cur_prev is not None:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                elif cur_prev is not None:
                    self.add(f"{cur_prev} --> {nid}")
                    
                self._register_entry(entry_holder, nid)   # [NEW]
                self.label_nodes[m_label.group(1)] = nid  # [NEW]

                cur_prev = nid
                any_node_created = True
                i = next_i
                continue
            

            # ---- if ----
            if lstrip.startswith("if") and not lstrip.startswith("else"):
                cur_prev, i = self._handle_if(lines, i, n, cur_prev, first_label, entry_holder)
                first_label = None
                any_node_created = True
                continue

            # ---- do { ... } while(...) ----
            # 'do'만 있는 줄 또는 'do {' 형태를 타겟
            if re.match(r"\bdo\b", lstrip):
                cur_prev, i = self._handle_do_while(lines, i, n, cur_prev, first_label, entry_holder)
                first_label = None
                any_node_created = True
                continue            

            # ---- while ----
            if lstrip.startswith("while"):
                cur_prev, i = self._handle_loop(lines, i, n, cur_prev, first_label, kind="while", entry_holder=entry_holder)
                first_label = None
                any_node_created = True
                continue

            # ---- for ----
            if lstrip.startswith("for"):
                cur_prev, i = self._handle_loop(lines, i, n, cur_prev, first_label, kind="for", entry_holder=entry_holder)
                first_label = None
                any_node_created = True
                continue

            # ---- switch ----
            if lstrip.startswith("switch"):
                cur_prev, i = self._handle_switch(lines, i, n, cur_prev, first_label, entry_holder)
                first_label = None
                any_node_created = True
                continue

            s_lower = stripped.lower()

            # ---- goto ----
            if s_lower.startswith("goto"):
                nid = self.nid()
                self._bind_node_line(nid, i)
                label_txt = self._clean_label(raw)
                self.add(f'{nid}["{label_txt}"]')

                # 이전 노드에서 goto 로의 엣지 (cur_prev 가 있을 때만)
                if first_label and cur_prev is not None:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                elif cur_prev is not None:
                    self.add(f"{cur_prev} --> {nid}")

                # label 이름 추출
                m = re.match(r"\s*goto\s+([A-Za-z_]\w*)", raw)
                target = m.group(1) if m else None
                self.goto_pending.append((nid, target))
                self.goto_nodes.add(nid)
                self._register_entry(entry_holder, nid)

                any_node_created = True
                i = next_i

                if not is_top_level:
                    # 블록 내부에서는 goto 를 '마지막 노드'로 취급하고 종료
                    cur_prev = nid
                    break
                else:
                    # top-level 에서는 이후 코드를 계속 스캔하지만
                    # 제어 흐름은 여기서 끊긴 것으로 본다.
                    cur_prev = None
                    continue

            # ---- break / continue 특별 처리 ----
            if s_lower.startswith("break"):
                nid = self.nid()
                self._bind_node_line(nid, i)
                label = self._clean_label(raw)
                self.add(f'{nid}["{label}"]')

                if first_label:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                else:
                    self.add(f"{cur_prev} --> {nid}")

                # 현재 가장 안쪽 루프에 break 노드 등록
                # 단, switch 안에 있을 때는 loop-break 로 취급하지 않는다.
                if self.switch_depth == 0 and self.break_stack:
                    self.break_stack[-1].append(nid)

                self._register_entry(entry_holder, nid)

                cur_prev = nid
                any_node_created = True
                i = next_i
                # break 이후 이 블록에서는 더 내려가지 않으므로 종료
                break

            if s_lower.startswith("continue"):
                nid = self.nid()
                self._bind_node_line(nid, i)
                label = self._clean_label(raw)
                self.add(f'{nid}["{label}"]')

                if first_label:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                else:
                    self.add(f"{cur_prev} --> {nid}")

                # 현재 가장 안쪽 루프에 continue 노드 등록
                if self.continue_stack:
                    self.continue_stack[-1].append(nid)

                cur_prev = nid
                any_node_created = True
                i = next_i
                # continue 이후 아래 코드는 실행되지 않으므로 종료
                break

            # ---- 그 외 단순 statement ----
            node_type = self._classify_simple(raw)
            nid = self.nid()
            self._bind_node_line(nid, i)
            label = self._clean_label(raw)

            if node_type == "terminator":
                # return 문 노드
                self.add(f'{nid}(["{label}"]):::term')

                if first_label and cur_prev is not None:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                elif cur_prev is not None:
                    self.add(f"{cur_prev} --> {nid}")
                    
                # return → end 노드 연결
                if self.end_node:
                    self.add(f"{nid} --> {self.end_node}")
                    cur_prev = self.end_node
                else:
                    cur_prev = nid

                any_node_created = True
                i = next_i
                # return 이후 이 블록에서 더 이상 실행 경로가 없으므로 종료
                break

            else:
                # 일반 statement
                self.add(f'{nid}["{label}"]')

                if first_label and cur_prev is not None:
                    self.add(f"{cur_prev} -->|{first_label}| {nid}")
                    first_label = None
                elif cur_prev is not None:
                    self.add(f"{cur_prev} --> {nid}")
                    
                cur_prev = nid
                any_node_created = True
                i = next_i

        if not any_node_created:
            # 아무 statement 가 없는 블록

            # 1) 첫 엣지에 라벨이 없는 경우 (예: for(;;) 의 본문이 비어 있음)
            # if first_label is None:
            #     return cur_prev

            # 2) if 의 True/False 처럼 라벨이 필요한 경우에만 pass 노드 생성
            # nid = self.nid()
            # self.add(f'{nid}["pass"]')
            # if cur_prev is not None:
            #     self.add(f"{cur_prev} -->|{first_label}| {nid}")
            # cur_prev = nid
            return cur_prev

        return cur_prev

    # ---- if / else / else if 처리 ----
    def _handle_if(self, lines, idx, end_idx, prev_node, edge_label, entry_holder):
        """
        if / else if / else 체인을 한 번에 처리.
        - if, else if 들은 각각 다이아몬드 노드
        - 마지막 else(있으면)는 일반 시퀀스로 파싱
        - 비어 있는 else 블록은 노드 없이 바로 merge로 연결
        - 멀티라인 조건식, 한 줄 if(if(cond) stmt;) 모두 지원
        """
        branches = []            # 각 if의 True 경로 마지막 노드들
        current_prev = prev_node
        current_edge = edge_label
        i = idx
        last_cond_id = None

        has_final_else = False
        final_else_idx = None
        after_idx = None


        # [ADD] _handle_if() 안쪽(while True 전에) 로컬 헬퍼 추가
        def _match_inline_brace_block(s: str):
            """
            한 줄 s 안에서 첫 '{' 기준으로 매칭되는 '}' 위치를 찾는다.
            같은 줄에서 outer 블록이 닫히면 (open_idx, close_idx, inner_text) 반환.
            못 찾으면 None.
            """
            if "{" not in s or "}" not in s:
                return None
            open_idx = s.find("{")
            depth = 0
            for k in range(open_idx, len(s)):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        close_idx = k
                        inner = s[open_idx + 1 : close_idx]
                        return open_idx, close_idx, inner
            return None

        def _inline_to_lines(inner: str) -> list[str]:
            """
            inline 블록 문자열을 파서가 읽을 수 있게 줄로 펼친다.
            ; { } 뒤에 개행을 강제로 넣어서 '정상 포맷'처럼 만든다.
            """
            out = []
            buf = []
            for ch in inner:
                buf.append(ch)
                if ch in [";", "{", "}"]:
                    line = "".join(buf).strip()
                    if line:
                        out.append(line)
                    buf = []
            tail = "".join(buf).strip()
            if tail:
                out.append(tail)
            return out
        

        while True:
            # ----- 조건 헤더(멀티라인) 수집 -----
            header_start = i
            header_lines = []
            paren_balance = 0
            j = i
            while j < end_idx:
                line = lines[j]
                header_lines.append(line)
                paren_balance += line.count("(") - line.count(")")
                # 괄호가 한번이라도 나왔고 다시 0 이하가 되면 조건식 끝으로 본다
                if paren_balance <= 0 and "(" in "".join(header_lines):
                    break
                j += 1
            header_end = j if j < end_idx else i

            first_line = lines[header_start]
            stripped_first = first_line.lstrip()
            is_else_if = stripped_first.startswith("else if")

            # 라벨 텍스트 구성
            header_text = " ".join(l.strip() for l in header_lines)
            if is_else_if:
                header_text = header_text.replace("else if", "if", 1)

            cond_text = self._extract_if_condition(header_text)
            cond_label = self._clean_cond_label(cond_text)
            cond_id = self.nid()
            last_cond_id = cond_id
            self.add(self._make_cond_node(cond_id, cond_label))
            self._bind_node_line(cond_id, header_start)
            self._register_entry(entry_holder, cond_id)

            # 이전 노드 → 현재 if
            if current_edge:
                self.add(f"{current_prev} -->|{current_edge}| {cond_id}")
            else:
                self.add(f"{current_prev} --> {cond_id}")

            # ----- then 블록(True) 위치 파악 -----
            brace_idx = None

            # 1) 현재 if/else if 헤더 라인에서 '{' 먼저 검색
            if "{" in lines[header_end]:
                brace_idx = header_end
            else:
                # 2) 헤더 이후의 첫 non-empty 라인을 찾고, 거기서 '{' 검색
                j = header_end + 1
                while j < end_idx and not lines[j].strip():
                    j += 1
                if j < end_idx and "{" in lines[j]:
                    brace_idx = j

            inline_stmt = None
            temp_lines = None

            # 3) 블록이 아닌 한 줄 if(if (cond) stmt;) 인지 검사
            #    (brace 블록이 잡힌 경우에는 inline if 로 보지 않도록 함)
            if brace_idx is None:
                cond_line_last = lines[header_end]
                m_inline = re.match(r"\s*(?:else\s+)?if\s*\([^)]*\)\s*(.+)", cond_line_last)
                if m_inline:
                    inline_stmt = m_inline.group(1).strip()
                    if inline_stmt and not inline_stmt.startswith("{"):
                        temp_lines = [inline_stmt]
                    else:
                        inline_stmt = None

            if brace_idx is not None:
                # ✅ [NEW] 같은 줄에 { ... } 가 닫히는 inline 블록이면 내용 유실 방지
                m_inline_blk = _match_inline_brace_block(lines[brace_idx])
                if m_inline_blk is not None:
                    _, _, inner = m_inline_blk
                    temp_lines = _inline_to_lines(inner)
                    then_start, then_end = 0, len(temp_lines)
                    after_then = brace_idx + 1
                else:
                    # 일반 { ... } 블록(여러 줄)
                    then_start, then_end, after_then = self._find_block(lines, brace_idx)
            elif temp_lines is not None:
                # if(cond) stmt; 형태 (중괄호 없음)
                then_start, then_end = 0, 1
                after_then = header_end + 1
            else:
                # 4) 중괄호도 없고 같은 줄에 statement 도 없으면
                #    헤더 다음의 첫 non-empty 라인을 then 으로 사용
                k = header_end + 1
                while k < end_idx and not lines[k].strip():
                    k += 1
                then_start, then_end, after_then = k, k + 1, k + 1


            # ----- True 분기 파싱 -----
            if temp_lines is not None:
                then_exit = self._parse_sequence(
                    temp_lines, 0, 1, cond_id, first_edge_label="True"
                )
            else:
                then_exit = self._parse_sequence(
                    lines, then_start, then_end, cond_id, first_edge_label="True"
                )

            if then_exit is not None and then_exit != self.end_node:
                branches.append(then_exit)

            # ----- 다음 토큰: else if / else / nothing -----
            k = after_then
            while k < end_idx and not lines[k].strip():
                k += 1

            if k >= end_idx:
                after_idx = after_then
                has_final_else = False
                break

            t = lines[k].lstrip()
            if t.startswith("else if"):
                # else if → 다시 루프
                current_prev = cond_id
                current_edge = "False"
                i = k
                continue

            if t.startswith("else"):
                has_final_else = True
                final_else_idx = k
                break

            # else / else if 둘 다 아니면 if 체인 종료
            after_idx = after_then
            has_final_else = False
            final_else_idx = None
            break

        # ----- 마지막 else 블록 처리 -----
        if has_final_else and final_else_idx is not None:
            brace_idx = None
            if "{" in lines[final_else_idx]:
                brace_idx = final_else_idx
            else:
                m = final_else_idx + 1
                while m < end_idx and not lines[m].strip():
                    m += 1
                if m < end_idx and "{" in lines[m]:
                    brace_idx = m

            else_temp_lines = None

            if brace_idx is not None:
                m_inline_blk = _match_inline_brace_block(lines[brace_idx])
                if m_inline_blk is not None:
                    _, _, inner = m_inline_blk
                    else_temp_lines = _inline_to_lines(inner)
                    else_start, else_end, after_else = 0, len(else_temp_lines), brace_idx + 1
                else:
                    else_start, else_end, after_else = self._find_block(lines, brace_idx)
            else:
                m = final_else_idx + 1
                while m < end_idx and not lines[m].strip():
                    m += 1
                else_start, else_end, after_else = m, m + 1, m + 1

            # else 블록이 비어 있는지 검사
            if else_start is not None and not _block_is_effectively_empty(
                lines, else_start, else_end
            ):
                if else_temp_lines is not None:
                    else_exit = self._parse_sequence(
                        else_temp_lines, 0, len(else_temp_lines), last_cond_id, first_edge_label="False"
                    )
                else:
                    else_exit = self._parse_sequence(
                        lines, else_start, else_end, last_cond_id, first_edge_label="False"
                    )

                # 실질적으로 다음으로 이어지는 분기들만 모음
                non_terminals = []
                for b in branches:
                    if (
                        not self._is_loop_control_node(b)
                        and b != self.end_node
                        and b not in self.goto_nodes        # [NEW]
                    ):
                        non_terminals.append(b)

                if (
                    else_exit is not None
                    and else_exit != self.end_node
                    and not self._is_loop_control_node(else_exit)
                    and else_exit not in self.goto_nodes     # [NEW]
                ):
                    non_terminals.append(else_exit)

                # 모두 종료 경로라면 merge 없이 끝
                if not non_terminals:
                    return (
                        self.end_node if self.end_node is not None else last_cond_id,
                        after_else,
                    )

                # 살아남는 경로가 있으면 merge 생성
                merge = self.nid()
                self.add(f'{merge}(["merge"]):::merge')
                for n in non_terminals:
                    self.add(f"{n} --> {merge}")

            else:
                # 내용 없는 else: False 는 바로 merge 로
                merge = self.nid()
                self.add(f'{merge}(["merge"]):::merge')
                for b in branches:
                    if not self._is_loop_control_node(b):
                        self.add(f"{b} --> {merge}")
                self.add(f"{last_cond_id} -->|False| {merge}")

            after_idx = after_else

        else:
            # else 자체가 없는 경우
            merge = self.nid()
            self.add(f'{merge}(["merge"]):::merge')
            for b in branches:
                if (
                    not self._is_loop_control_node(b)
                    and b not in self.goto_nodes
                    and b != self.end_node
                ):
                    self.add(f"{b} --> {merge}")
            self.add(f"{last_cond_id} -->|False| {merge}")

        return merge, after_idx

    # ---- while / for 루프 처리 ----
    def _handle_loop(self, lines, idx, end_idx, prev_node, edge_label, kind="while", entry_holder=None):
        # ✅ for/while 헤더가 여러 줄로 끊긴 경우(if처럼) ')' 닫힐 때까지 합친다
        header_lines = []
        paren_balance = 0
        j = idx
        saw_paren = False

        while j < end_idx:
            line = lines[j]
            header_lines.append(line)
            if "(" in line:
                saw_paren = True
            paren_balance += line.count("(") - line.count(")")
            # 괄호를 한 번이라도 봤고, balance가 0이 되면 헤더 종료
            if saw_paren and paren_balance <= 0:
                break
            # 단일라인 헤더(괄호가 없는 for(;;) 같은 케이스)는 여기서 끝내지 않음
            j += 1

        header_end = j if j < end_idx else idx
        loop_line = " ".join([x.strip() for x in header_lines])  # ✅ 헤더 통합본
        loop_label_full = self._clean_label(loop_line)

        # ---------- for 문 처리 ----------
        if kind == "for":
            # 헤더 파싱: for (init; cond; post)
            m = re.search(r"\bfor\s*\(", loop_line)
            init = cond = post = ""
            if m:
                # ✅ for( ... ) 괄호 내용만 depth로 정확히 추출
                k = m.end()
                depth = 1
                buf = []
                while k < len(loop_line) and depth > 0:
                    ch = loop_line[k]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    buf.append(ch)
                    k += 1
                inside = "".join(buf)

                # ✅ top-level ';'만 분리 (괄호 안 ';' 방지)
                parts = []
                cur = []
                depth2 = 0
                for ch in inside:
                    if ch == "(":
                        depth2 += 1
                    elif ch == ")":
                        depth2 = max(0, depth2 - 1)

                    if ch == ";" and depth2 == 0:
                        parts.append("".join(cur))
                        cur = []
                    else:
                        cur.append(ch)
                parts.append("".join(cur))

                if len(parts) >= 1:
                    init = parts[0].strip()
                if len(parts) >= 2:
                    cond = parts[1].strip()
                if len(parts) >= 3:
                    post = parts[2].strip()


            # 먼저 본문 블록 위치부터 찾는다
            brace_idx = None
            if "{" in loop_line:
                brace_idx = idx
            else:
                j = idx + 1
                while j < end_idx and not lines[j].strip():
                    j += 1
                if j < end_idx and "{" in lines[j]:
                    brace_idx = j

            if brace_idx is not None:
                body_start, body_end, after_idx = self._find_block(lines, brace_idx)
            else:
                j = idx + 1
                while j < end_idx and not lines[j].strip():
                    j += 1
                body_start, body_end, after_idx = j, j + 1, j + 1

            # 완전 빈 for(;;) 루프면 아예 없다고 보고 넘어간다
            if (not cond) and _block_is_effectively_empty(lines, body_start, body_end):
                return prev_node, after_idx

            cur_prev = prev_node
            cond_edge_label = edge_label

            # init 문장 노드 (있을 때만)
            if init:
                init_id = self.nid()
                init_label = self._clean_label(init + ";")
                self.add(f'{init_id}["{init_label}"]')

                self._bind_node_line(init_id, idx)
                
                if edge_label:
                    self.add(f"{prev_node} -->|{edge_label}| {init_id}")
                else:
                    self.add(f"{prev_node} --> {init_id}")

                self._register_entry(entry_holder, init_id)
                cur_prev = init_id
                cond_edge_label = None  # 조건으로 가는 엣지는 라벨 X

            # 이 for 루프 안에서 나오는 break/continue 를 수집하기 위해 스택 push
            self.break_stack.append([])
            self.continue_stack.append([])

            # 조건 다이아몬드
            cond_id = self.nid()
            cond_text = cond if cond else "for(;;)"
            cond_label = self._clean_cond_label(cond_text)
            self.add(self._make_cond_node(cond_id, cond_label))
            self._bind_node_line(cond_id, idx)
            self._register_entry(entry_holder, cond_id)

            if cond_edge_label:
                self.add(f"{cur_prev} -->|{cond_edge_label}| {cond_id}")
            else:
                self.add(f"{cur_prev} --> {cond_id}")

            # --- 조건이 있는 일반 for ---
            if cond:
                body_exit = self._parse_sequence(
                    lines, body_start, body_end, cond_id, first_edge_label="True"
                )
                break_nodes = self.break_stack.pop()
                continue_nodes = self.continue_stack.pop()

                # post (증감식)
                if post:
                    post_id = self.nid()
                    post_label = self._clean_label(post + ";")
                    self.add(f'{post_id}["{post_label}"]')

                    self._bind_node_line(post_id, idx)

                    if (
                        body_exit is not None
                        and body_exit != self.end_node
                        and body_exit not in break_nodes
                        and body_exit not in continue_nodes
                        and body_exit not in self.goto_nodes
                    ):
                        self.add(f"{body_exit} --> {post_id}")

                    # continue 는 post 로 점프
                    for c in continue_nodes:
                        self.add(f"{c} --> {post_id}")

                    self.add(f"{post_id} --> {cond_id}")
                else:
                    # post 가 없으면 body 및 continue 가 직접 조건으로
                    if (
                        body_exit is not None
                        and body_exit != self.end_node
                        and body_exit not in break_nodes
                        and body_exit not in continue_nodes
                        and body_exit not in self.goto_nodes
                    ):
                        self.add(f"{body_exit} --> {cond_id}")
                    for c in continue_nodes:
                        self.add(f"{c} --> {cond_id}")

                # for 탈출 노드
                after_node = self.nid()
                self.add(f'{after_node}(["after for"])')

                # cond 가 거짓일 때
                self.add(f"{cond_id} -->|False| {after_node}")
                # break 들도 after for 로
                for b in break_nodes:
                    self.add(f"{b} --> {after_node}")

                return after_node, after_idx

            # --- for(;;) : 조건 없는 무한루프 ---
            self.infinite_loop_nodes.add(cond_id)
            body_exit = self._parse_sequence(
                lines, body_start, body_end, cond_id, first_edge_label=None
            )
            break_nodes = self.break_stack.pop()
            continue_nodes = self.continue_stack.pop()

            if post:
                post_id = self.nid()
                post_label = self._clean_label(post + ";")
                self.add(f'{post_id}["{post_label}"]')

                self._bind_node_line(post_id, idx)

                if (
                    body_exit is not None
                    and body_exit != self.end_node
                    and body_exit not in break_nodes
                    and body_exit not in continue_nodes
                    and body_exit not in self.goto_nodes
                ):
                    self.add(f"{body_exit} --> {post_id}")

                for c in continue_nodes:
                    self.add(f"{c} --> {post_id}")

                self.add(f"{post_id} --> {cond_id}")
            else:
                if (
                    body_exit is not None
                    and body_exit != self.end_node
                    and body_exit not in break_nodes
                    and body_exit not in continue_nodes
                    and body_exit not in self.goto_nodes
                ):
                    self.add(f"{body_exit} --> {cond_id}")
                for c in continue_nodes:
                    self.add(f"{c} --> {cond_id}")

            # 무한루프지만 break 가 있으면 그 경로만 after for 로 빼준다
            if break_nodes:
                after_node = self.nid()
                self.add(f'{after_node}(["after for"])')
                for b in break_nodes:
                    self.add(f"{b} --> {after_node}")
                return after_node, after_idx

            # break 도 없으면 이론상 끝나지 않는 루프이므로 cond_id 반환
            return cond_id, after_idx

        # ---------- while 문 처리 ----------
        # ✅ if처럼 'while ( ... )' 까지만 추출 (멀티라인도 loop_line에 합쳐져 있음)
        m = re.search(r"\bwhile\s*\(", loop_line)
        cond_text = loop_line
        if m:
            k = m.end()
            depth = 1
            while k < len(loop_line) and depth > 0:
                ch = loop_line[k]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                k += 1
            if depth == 0:
                cond_text = loop_line[:k]  # 'while ( ... )' 까지만

        loop_id = self.nid()
        loop_label = self._clean_cond_label(cond_text)
        
        self.add(self._make_cond_node(loop_id, loop_label))
        self._bind_node_line(loop_id, idx)
        self._register_entry(entry_holder, loop_id)

        if prev_node is not None:
            if edge_label:
                self.add(f"{prev_node} -->|{edge_label}| {loop_id}")
            else:
                self.add(f"{prev_node} --> {loop_id}")
        
        brace_idx = None
        if "{" in loop_line:
            brace_idx = idx
        else:
            j = idx + 1
            while j < end_idx and not lines[j].strip():
                j += 1
            if j < end_idx and "{" in lines[j]:
                brace_idx = j

        if brace_idx is not None:
            body_start, body_end, after_idx = self._find_block(lines, brace_idx)
        else:
            j = idx + 1
            while j < end_idx and not lines[j].strip():
                j += 1
            body_start, body_end, after_idx = j, j + 1, j + 1

        # 이 while 안에서 나오는 break/continue 를 수집하기 위해 스택 push
        self.break_stack.append([])
        self.continue_stack.append([])            

        body_exit = self._parse_sequence(
            lines, body_start, body_end, loop_id, first_edge_label="True"
        )
        
        # 해당 while 루프에서 수집된 break/continue 노드
        break_nodes = self.break_stack.pop()
        continue_nodes = self.continue_stack.pop()

        # 정상적으로 루프 바디를 빠져나온 경로만 조건으로 되돌리기
        if (
            body_exit is not None
            and body_exit != self.end_node
            and body_exit not in break_nodes
            and body_exit not in self.goto_nodes        # [추가]
        ):
            self.add(f"{body_exit} --> {loop_id}")

        # continue 는 항상 다시 while 조건으로
        for c in continue_nodes:
            self.add(f"{c} --> {loop_id}")

        # while 을 빠져나가는 노드
        after_node = self.nid()
        self.add(f'{after_node}(["after while"])')

        # 조건이 거짓일 때
        self.add(f"{loop_id} -->|False| {after_node}")
        # break; 로 나오는 경로
        for b in break_nodes:
            self.add(f"{b} --> {after_node}")

        return after_node, after_idx

    # ---- do { ... } while(...) 처리 ----
    def _handle_do_while(self, lines, idx, end_idx, prev_node, edge_label, entry_holder):
        """
        do {
            ...
        } while (cond);
        를 처리하는 핸들러.

        구조:
        prev → do_entry → (body ...) → cond → True → do_entry, False → after do-while
        break;     → after do-while
        continue;  → cond
        """
        do_line = lines[idx]

        # do-entry 노드 (body 시작점 역할)
        do_id = self.nid()
        self.add(f'{do_id}["do"]')

        self._bind_node_line(do_id, idx)

        if edge_label:
            self.add(f"{prev_node} -->|{edge_label}| {do_id}")
        else:
            self.add(f"{prev_node} --> {do_id}")

        self._register_entry(entry_holder, do_id)

        # --- body 블록 위치 찾기 (do { ... } while(...); ) ---
        brace_idx = None
        if "{" in do_line:
            brace_idx = idx
        else:
            j = idx + 1
            while j < end_idx and not lines[j].strip():
                j += 1
            if j < end_idx and "{" in lines[j]:
                brace_idx = j

        if brace_idx is not None:
            body_start, body_end, after_body = self._find_block(lines, brace_idx)
        else:
            # 중괄호 없는 한 줄짜리 do 문 (do stmt; while(...);)
            j = idx + 1
            while j < end_idx and not lines[j].strip():
                j += 1
            body_start, body_end, after_body = j, j + 1, j + 1

        # --- while 꼬리(조건) 위치 찾기 ---
        # 1) "} while (cond);" 처럼 '}'와 while 이 같은 줄에 있는 경우 먼저 처리
        cond_line_idx = None
        if body_end > body_start:
            last_idx = body_end - 1
            last_line = lines[last_idx]
            # '}' 가 있고, 그 라인 안에 while(...) 도 같이 있으면 tail 로 간주
            if "}" in last_line and "while" in last_line:
                cond_line_idx = last_idx
                # body 에서는 이 라인을 제외 (순수한 바디만 남김)
                body_end = last_idx

        # 2) 일반적인 형태: 블록 뒤 다음 줄에 "while (cond);" 이 오는 경우
        if cond_line_idx is None:
            k = after_body
            while k < end_idx and not lines[k].strip():
                k += 1
            if k < end_idx and lines[k].lstrip().startswith("while"):
                cond_line_idx = k

        # 3) 그래도 while 꼬리를 못 찾으면, 그냥 평범한 do 블록처럼 처리
        if cond_line_idx is None:
            body_exit = self._parse_sequence(
                lines, body_start, body_end, do_id, first_edge_label=None
            )
            # 이 경우에는 루프 구조가 아니라고 본다.
            return body_exit, after_body

        # --- 여기서부터는 do-while 을 진짜 루프로 처리 ---
        cond_line = lines[cond_line_idx]

        # 이 do-while 안에서 나오는 break/continue 수집
        self.break_stack.append([])
        self.continue_stack.append([])

        # body 파싱
        body_exit = self._parse_sequence(
            lines, body_start, body_end, do_id, first_edge_label=None
        )

        break_nodes = self.break_stack.pop()
        continue_nodes = self.continue_stack.pop()

        # 조건 다이아몬드 노드
        loop_id = self.nid()
        loop_label = self._clean_cond_label(cond_line)
        self.add(self._make_cond_node(loop_id, loop_label))
        self._bind_node_line(loop_id, cond_line_idx)

        # 정상적으로 body 를 빠져나온 경로만 cond 로 연결
        if (
            body_exit is not None
            and body_exit != self.end_node
            and body_exit not in break_nodes
            and body_exit not in continue_nodes
            and body_exit not in self.goto_nodes        # [추가]
        ):
            self.add(f"{body_exit} --> {loop_id}")

        # continue 는 cond 로 점프
        for c in continue_nodes:
            self.add(f"{c} --> {loop_id}")

        # cond 의 True: 다시 do-entry 로
        self.add(f"{loop_id} -->|True| {do_id}")

        # do-while 을 빠져나가는 노드
        after_node = self.nid()
        self.add(f'{after_node}(["after do-while"])')

        # cond 의 False: after do-while
        self.add(f"{loop_id} -->|False| {after_node}")

        # break 들도 after do-while 로
        for b in break_nodes:
            self.add(f"{b} --> {after_node}")

        # while 꼬리 다음 줄부터 계속 파싱
        after_idx = cond_line_idx + 1
        # (보통 cond_line_idx+1 과 after_body 는 같지만, 방어적으로 cond 기준 사용)
        return after_node, after_idx


    # ---- switch / case 처리 ----
    def _handle_switch(self, lines, idx, end_idx, prev_node, edge_label, entry_holder):
        sw_line = lines[idx]
        sw_label = self._clean_cond_label(sw_line)
        sw_id = self.nid()
        self.add(self._make_cond_node(sw_id, sw_label))
        self._bind_node_line(sw_id, idx)

        # switch 진입점 기록
        self._register_entry(entry_holder, sw_id)

        # 이전 노드에서 switch 다이아몬드로 연결
        if edge_label:
            self.add(f"{prev_node} -->|{edge_label}| {sw_id}")
        else:
            self.add(f"{prev_node} --> {sw_id}")

        # ----- switch(...) { ... } 블록 찾기 -----
        brace_idx = None
        if "{" in sw_line:
            brace_idx = idx
        else:
            j = idx + 1
            while j < end_idx and not lines[j].strip():
                j += 1
            if j < end_idx and "{" in lines[j]:
                brace_idx = j

        if brace_idx is not None:
            body_start, body_end, after_idx = self._find_block(lines, brace_idx)
        else:
            body_start, body_end, after_idx = idx + 1, idx + 1, idx + 1

        self.switch_depth += 1
        try:
            # ----- case / default 헤더 라인 인덱스 수집 -----
            header_idxs = []
            i = body_start
            while i < body_end:
                line = lines[i]
                # 앞뒤 공백, 탭 등 무시하고 case/default 인지만 본다
                if re.match(r"^\s*case\b", line) or re.match(r"^\s*default\b", line):
                    header_idxs.append(i)
                i += 1

            if not header_idxs:
                single = self.nid()
                self.add(f'{single}["(switch body)"]')
                self.add(f"{sw_id} --> {single}")
                return single, after_idx

            # 공통 merge 노드 (after switch)
            merge = self.nid()
            self.add(f'{merge}(["after switch"]):::merge')

            # 각 case/header 라인에 대한 노드 먼저 생성
            case_nodes = {}
            for h in header_idxs:
                case_header_line = lines[h]
                case_id = self.nid()
                case_label = self._clean_label(case_header_line.strip())
                self.add(f'{case_id}["{case_label}"]')
                self._bind_node_line(case_id, h)
                self.add(f"{sw_id} --> {case_id}")
                case_nodes[h] = case_id

            # ----- fall-through 그룹 계산 (break/return/goto 기준) -----
            groups = []
            pos = 0
            while pos < len(header_idxs):
                group = [header_idxs[pos]]
                j = pos + 1

                while j < len(header_idxs):
                    prev_h = header_idxs[j - 1]
                    next_h = header_idxs[j]

                    # prev_h 와 next_h 사이에 break/return/goto 가 있으면
                    # fall-through 가 끊긴 것으로 본다.
                    has_stop = False
                    for k in range(prev_h + 1, next_h):
                        s = lines[k].strip()
                        if s.startswith(("break", "return", "goto")):
                            has_stop = True
                            break

                    if has_stop:
                        break

                    group.append(next_h)
                    j += 1

                groups.append(group)
                pos = j

            # 다음 header 위치 빠르게 찾는 맵
            idx_to_next = {}
            for i, h in enumerate(header_idxs):
                nxt = header_idxs[i + 1] if i + 1 < len(header_idxs) else body_end
                idx_to_next[h] = nxt

            # ----- 각 그룹별로 body 파싱 -----
            for group in groups:
                # 그룹의 마지막 header(실제 실행 body가 붙는 case)
                main_header = group[-1]
                main_case_id = case_nodes[main_header]

                # 대표 case의 본문: main_header 이후 ~ 다음 header(or switch 끝)
                case_body_start = main_header + 1
                case_body_end = idx_to_next[main_header]

                # 앞쪽 공백 / { / } 제거
                while case_body_start < case_body_end:
                    s = lines[case_body_start].strip()
                    if not s or s in ("{", "}"):
                        case_body_start += 1
                    else:
                        break

                # 내용이 사실상 없는 그룹이면 각 case 를 그냥 merge 로 보냄
                if case_body_start >= case_body_end or _block_is_effectively_empty(
                    lines, case_body_start, case_body_end
                ):
                    for h in group:
                        self.add(f"{case_nodes[h]} --> {merge}")
                    continue

                # 1) 대표 case(마지막 header)의 본문 파싱
                entry_holder_body = []
                exit_node = self._parse_sequence(
                    lines,
                    case_body_start,
                    case_body_end,
                    main_case_id,
                    first_edge_label=None,
                    entry_holder=entry_holder_body,
                )
                entry_node = entry_holder_body[0] if entry_holder_body else main_case_id

                # 2) 같은 그룹의 "앞쪽" case 들에 대한 fall-through 처리
                for idx_g in range(len(group) - 2, -1, -1):  # 뒤에서부터 순회
                    h = group[idx_g]
                    next_h = group[idx_g + 1]

                    cid = case_nodes[h]
                    next_case_id = case_nodes[next_h]

                    sub_start = h + 1
                    sub_end = next_h

                    # 공백 / { / }만 앞에서 제거
                    while sub_start < sub_end:
                        s = lines[sub_start].strip()
                        if not s or s in ("{", "}"):
                            sub_start += 1
                        else:
                            break

                    # 사이에 실행 코드가 없으면 바로 다음 case 라벨로 연결
                    if sub_start >= sub_end or _block_is_effectively_empty(lines, sub_start, sub_end):
                        self.add(f"{cid} --> {next_case_id}")
                        continue

                    # 실행 코드가 있으면 별도 시퀀스로 파싱
                    sub_entry_holder = []
                    sub_exit = self._parse_sequence(
                        lines,
                        sub_start,
                        sub_end,
                        cid,
                        first_edge_label=None,
                        entry_holder=sub_entry_holder,
                    )

                    # 살아 있는 경로만 다음 case 라벨로 연결
                    if (
                        sub_exit is not None
                        and not self._is_loop_control_node(sub_exit)
                        and sub_exit not in self.goto_nodes
                        and sub_exit != self.end_node
                    ):
                        self.add(f"{sub_exit} --> {next_case_id}")

                # 3) 대표 case body 끝에서 merge 로 연결 (예: break; 등)
                if (
                    exit_node is not None
                    and not self._is_loop_control_node(exit_node)
                    and exit_node not in self.goto_nodes
                ):
                    self.add(f"{exit_node} --> {merge}")

            return merge, after_idx

        finally:
            self.switch_depth -= 1


def generate_flowchart_from_file(path: str, func_name: str, branch_shape: str = "rounded", macros: dict | None = None) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    code = p.read_text(encoding="utf-8", errors="ignore")
    func_name_clean = sanitize_func_name(func_name)   # ✅ 추가
    body = extract_function_body(code, func_name_clean, macros=macros or {})

    emitter = StructuredFlowEmitter(
        func_name_clean,
        branch_shape=branch_shape,
        macros=macros or {},
    )
    return emitter.emit_from_body(body)


# ---------------------------------------------------------------------------
# 파싱/그래프 (callgraph)
# ---------------------------------------------------------------------------

def parse_tu(path, clang_args):
    """
    선택한 C 파일이 있는 상위 폴더 전체를 전처리해서 *_clean 트리를 만들고,
    그 클린 트리 기준으로 libclang 파싱.
    """
    global _ci
    cindex = _ensure_libclang()
    _ci = cindex

    src_file = Path(path).resolve()
    if not src_file.exists():
        print(f"입력 파일 없음: {src_file}", file=sys.stderr)
        sys.exit(1)

    src_root = src_file.parent
    out_root = src_root.parent / (src_root.name + "_clean")

    preprocess_tree_vendor(str(src_root), str(out_root))

    clean_file = out_root / src_file.name
    if not clean_file.exists():
        try:
            rel = src_file.relative_to(src_root)
            clean_file = out_root / rel
        except ValueError:
            clean_file = src_file  # fallback

    clean_inc = collect_include_dirs(str(out_root))
    clang_args = clean_inc + clang_args

    parse_target = str(clean_file if clean_file.exists() else src_file)
    if not os.path.exists(parse_target):
        print(f"입력 파일 없음(클린/원본 모두 실패): {parse_target}", file=sys.stderr)
        sys.exit(1)

    idx = cindex.Index.create()
    try:
        tu = idx.parse(parse_target, args=clang_args)
    except cindex.TranslationUnitLoadError as e:
        print("Translation Unit 로드 실패: ", e, file=sys.stderr)
        sys.exit(1)

    errs = [d for d in tu.diagnostics if d.severity >= _ci.Diagnostic.Error]
    if errs:
        print(f"# libclang 진단: 에러 {len(errs)}개 (상세 출력 생략)", file=sys.stderr)
    return tu

def walk_calls(fn_cursor):
    callees = set()
    q = deque([fn_cursor])
    while q:
        cur = q.popleft()
        for ch in cur.get_children():
            if ch.kind == _ci.CursorKind.CALL_EXPR:
                name = node_spelling(ch.referenced) if ch.referenced else node_spelling(ch)
                if name:
                    callees.add(name)
            q.append(ch)
    return callees

def collect_functions(tu):
    fns = []
    for c in tu.cursor.get_children():
        if c.kind == _ci.CursorKind.FUNCTION_DECL and c.is_definition():
            fns.append(c)
    return fns

def build_callgraph(tu):
    fns = collect_functions(tu)
    graph = defaultdict(set)
    for f in fns:
        caller = node_spelling(f)
        for callee in walk_calls(f):
            graph[caller].add(callee)
    return graph, fns

# ---------------------------------------------------------------------------
# Mermaid 출력 (callgraph)
# ---------------------------------------------------------------------------

def print_callgraph_mermaid(graph):
    print("graph LR")
    if not graph:
        print('    Empty["<no functions>"]')
    for caller, callees in graph.items():
        if not callees:
            print(f'    {safe(caller)}["{caller}"]')
        for callee in sorted(callees):
            print(f'    {safe(caller)}["{caller}"] --> {safe(callee)}["{callee}"]')

# ---------------------------------------------------------------------------
# 시퀀스(로그)
# ---------------------------------------------------------------------------

def parse_sequence_log(log_path):
    events = []
    parts = set()
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("CALL"):
                try:
                    rest = line.split(" ", 1)[1].strip()
                    caller, callee = rest.split("->")
                    caller = caller.strip()
                    callee = callee.strip()
                    events.append((caller, callee))
                    parts.add(caller)
                    parts.add(callee)
                except Exception:
                    pass
    return events, sorted(parts)

def print_sequence_mermaid(events, participants):
    print("sequenceDiagram")
    if not participants:
        print("    Note over User: no events")
        return
    for p in participants:
        print(f"    participant {safe(p)} as {p}")
    for caller, callee in events:
        print(f"    {safe(caller)}->>{safe(callee)}: call")

# ---------------------------------------------------------------------------
# CLI 파서
# ---------------------------------------------------------------------------

def build_argparser():
    ap = argparse.ArgumentParser(
        description="C flowchart / callgraph / sequence (Mermaid)"
    )
    sub = ap.add_subparsers(dest="cmd", required=False)

    # 자동 변환 모드 추가
    p_auto = sub.add_parser("auto", help="폴더 전체 자동 변환(.c → mmd & png)")
    p_auto.add_argument("path", help="대상 폴더 경로")

    p_fc = sub.add_parser("flowchart", help="함수 플로우차트 생성 (Mermaid)")
    p_fc.add_argument("path", help="대상 C 소스 파일")
    p_fc.add_argument("--function", help="특정 함수만 (기본: main)")

    p_cg = sub.add_parser("callgraph", help="정적 호출 그래프 (Mermaid, libclang 필요)")
    p_cg.add_argument("path", help="대상 C 소스 파일")
    p_cg.add_argument("--clang-arg", action="append", default=[], help="libclang 전달 인자")

    p_sq = sub.add_parser("sequence", help="런타임 로그(txt)로 시퀀스 다이어그램 생성")
    p_sq.add_argument("log", help="로그 파일 경로")

    return ap

# ---------------------------------------------------------------------------
# 인터랙티브 모드
# ---------------------------------------------------------------------------

def _try_import_tk():
    try:
        import tkinter as tk
        from tkinter import filedialog
        return tk, filedialog
    except Exception:
        return None, None

def interactive_main():
    tk, filedialog = _try_import_tk()
    print("\n=== Auto Diagram (C) — Interactive ===")
    print("1) flowchart  2) callgraph  3) sequence")
    mode = (input("모드를 선택하세요 [1/2/3] (기본:1): ").strip() or "1")
    if mode not in ("1", "2", "3"):
        print("잘못된 입력입니다.")
        return

    # 1) flowchart : 미니 파서 버전 (libclang 사용 안 함)
    if mode == "1":
        path = None
        if tk:
            try:
                root_win = tk.Tk()
                root_win.withdraw()
                path = filedialog.askopenfilename(
                    title="C 소스 파일 선택",
                    filetypes=[("C source", "*.c"), ("All", "*.*")],
                )
            except Exception:
                path = None
        if not path:
            path = input("C 파일 경로: ").strip()
        if not path:
            print("파일이 선택되지 않았습니다.")
            return

        func = input("플로우차트를 만들 함수명? (기본: main): ").strip() or "main"

        try:
            mermaid = generate_flowchart_from_file(path, func)
            print(mermaid)
        except Exception as e:
            print(f"# 플로우차트 생성 실패: {e}", file=sys.stderr)
        return

    # 2, 3 번은 기존 방식 (callgraph/sequence)

    print("\n프리셋을 선택하세요:")
    print("1) MSVC(Visual Studio/Windows SDK)")
    print("2) MinGW(MSYS2)")
    print("3) None(기본값만)")
    preset_sel = (input("프리셋 [1/2/3] (기본:1): ").strip() or "1")
    if preset_sel == "1":
        base_args = preset_msvc()
    elif preset_sel == "2":
        base_args = preset_mingw()
    else:
        base_args = preset_none()

    auto_inc = (
        input("상위 폴더를 지정해 하위 전체를 자동 -I 등록할까요? [Y/n] (기본:Y): ")
        .strip()
        or "Y"
    ).lower() != "n"
    extra_inc = []
    if auto_inc:
        root = None
        if tk:
            try:
                root_win = tk.Tk()
                root_win.withdraw()
                root = filedialog.askdirectory(title="상위 폴더 선택")
            except Exception:
                root = None
        if not root:
            root = input("상위 폴더 경로 (엔터=현재 폴더): ").strip() or os.getcwd()
        if os.path.isdir(root):
            extra_inc = collect_include_dirs(root)
            print(f"[자동 include] {len(extra_inc)}개 경로 등록")
        else:
            print("상위 폴더가 올바르지 않습니다. 자동 include 생략.")

    user_args_str = input(
        "추가 인자(쉼표 구분, 예: -DMYFLAG=1,-D__EMBEDDED__): "
    ).strip()
    user_args = (
        [s.strip() for s in user_args_str.split(",") if s.strip()]
        if user_args_str
        else []
    )

    clang_args = base_args + extra_inc + user_args
    clang_args += [
        "-ferror-limit=0",
        "-Wno-implicit-function-declaration",
    ]

    if mode == "2":  # callgraph
        path = None
        if tk:
            try:
                root_win = tk.Tk()
                root_win.withdraw()
                path = filedialog.askopenfilename(
                    title="C 소스 파일 선택",
                    filetypes=[("C source", "*.c"), ("All", "*.*")],
                )
            except Exception:
                path = None
        if not path:
            path = input("C 파일 경로: ").strip()
        if not path:
            print("파일이 선택되지 않았습니다.")
            return

        tu = parse_tu(path, clang_args)
        graph, _ = build_callgraph(tu)
        print_callgraph_mermaid(graph)
        return

    # mode == "3" : sequence
    log_path = None
    if tk:
        try:
            root_win = tk.Tk()
            root_win.withdraw()
            log_path = filedialog.askopenfilename(
                title="시퀀스 로그 선택",
                filetypes=[("Text", "*.txt"), ("All", "*.*")],
            )
        except Exception:
            log_path = None
    if not log_path:
        log_path = input("로그 파일 경로: ").strip()
    if not log_path:
        print("파일이 선택되지 않았습니다.")
        return
    events, parts = parse_sequence_log(log_path)
    print_sequence_mermaid(events, parts)


def auto_generate_all(src_root: str):
    """
    src_root 아래 모든 .c 파일을 찾아서,
    각 함수별로 .mmd 및 .png 자동 생성.
    출력은 ./Out 폴더
    """
    src_root = Path(src_root).resolve()
    out_dir = Path.cwd() / "Out"
    out_dir.mkdir(exist_ok=True)

    print(f"[INFO] Source root: {src_root}")
    print(f"[INFO] Output dir : {out_dir}")

    # 모든 C 파일 탐색
    c_files = list(src_root.rglob("*.c"))
    if not c_files:
        print("[WARN] No C files found.")
        return

    for cf in c_files:
        print(f"\n[FILE] {cf}")

        code = cf.read_text(encoding="utf-8", errors="ignore")
        fnames = extract_function_names(code)

        if not fnames:
            print("  └─ No functions detected.")
            continue

        print(f"  └─ Functions found: {fnames}")

        for fn in fnames:
            try:
                mermaid = generate_flowchart_from_file(str(cf), fn)

                # 파일명 안전하게
                safe_name = re.sub(r"[^A-Za-z0-9_]", "_", fn)

                mmd_file = out_dir / f"{safe_name}.mmd"
                # png_file = out_dir / f"{safe_name}.png"
                svg_file = out_dir / f"{safe_name}.svg"

                # mmd 저장
                mmd_file.write_text(mermaid, encoding="utf-8")
                print(f"      - MMD saved: {mmd_file.name}")

                # PNG 생성 (mermaid-cli)
                try:
                    subprocess.run(
                        ["mmdc", "-i", str(mmd_file), "-o", str(svg_file)],
                        check=True
                    )
                    print(f"      - SVG saved: {svg_file.name}")
                except Exception as e:
                    print(f"      ! SVG generation failed: {e}")

            except Exception as e:
                print(f"      ! Error processing function {fn}: {e}")

# ---------------------------------------------------------------------------
# main  (IDE에서 실행할 때: 폴더 선택 → 함수별 MMD + SVG 생성)
# ---------------------------------------------------------------------------

def main():
    """
    IDE에서 실행하면 폴더를 선택하게 하고,
    그 폴더 아래의 모든 .c 파일을 찾아서
    함수별로 .mmd + .svg 를 ./out 폴더에 생성한다.
    """
    import tkinter as tk
    from tkinter import filedialog

    print("=== Auto Mermaid Generator (C Source) ===")
    print("생성할 C 소스 폴더를 선택하세요 ...")

    # 폴더 선택 GUI
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="C 소스 폴더 선택")

    if not folder:
        print("폴더가 선택되지 않았습니다. 종료합니다.")
        return

    src_root = Path(folder).resolve()
    out_dir = Path.cwd() / "out"   # 파이썬 파일이 있는 곳의 ./out
    out_dir.mkdir(exist_ok=True)

    print(f"\n[선택된 폴더] {src_root}")
    print(f"[출력 폴더] {out_dir}")

    # 폴더 내 모든 C 파일 탐색
    c_files = list(src_root.rglob("*.c"))
    if not c_files:
        print("C 파일(*.c)이 없습니다. 종료합니다.")
        return

    print(f"\n총 {len(c_files)}개의 C 파일을 찾았습니다.")
    print("변환 시작...\n")

    for cfile in c_files:
        print(f"▶ 파일 처리: {cfile}")
        try:
            code = cfile.read_text(encoding="utf-8", errors="ignore")
            funcs = extract_function_names(code)

            if not funcs:
                print("   - 함수 없음 (skip)")
                continue

            for fn in funcs:
                try:
                    mermaid = generate_flowchart_from_file(str(cfile), fn)

                    # 함수명에 파일 이름도 붙여서 충돌 방지
                    base = cfile.stem
                    safe_fn = re.sub(r"[^A-Za-z0-9_]", "_", fn)
                    mmd_path = out_dir / f"{base}__{safe_fn}.mmd"
                    svg_path = out_dir / f"{base}__{safe_fn}.svg"

                    # .mmd 파일 저장
                    mmd_path.write_text(mermaid, encoding="utf-8")

                    # SVG 변환 (mermaid-cli 필요: mmdc)
                    # cmd.exe를 써서 PATH 안의 mmdc를 찾도록 함
                    cmd = f'mmdc -i "{mmd_path}" -o "{svg_path}"'
                    ret = os.system(cmd)

                    if ret == 0:
                        print(f"   - {fn}(): OK (SVG)")
                    else:
                        print(f"   - {fn}(): mmdc 실행 실패 (ret={ret})")

                except Exception as e:
                    print(f"   - {fn}(): 실패 → {e}")

        except Exception as e:
            print(f"[파일 오류] {cfile}: {e}")

    print("\n=== 모든 작업 완료! ===")


if __name__ == "__main__":
    main()

