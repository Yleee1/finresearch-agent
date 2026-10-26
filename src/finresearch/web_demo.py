from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .bm25 import BM25Retriever
from .budgets import ResearchBudget
from .evidence import EvidencePackBuilder
from .io import load_documents
from .query_analysis import QueryAnalyzer
from .retrieval_critic import RetrievalCritic
from .retrieval_value_gate import RetrievalValueGate
from .schemas import Document

_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinResearch Evidence Explorer</title>
<style>
:root{color-scheme:dark;--bg:#07101d;--panel:#101d2d;--line:#29415c;--text:#edf5ff;--muted:#9fb4c9;--accent:#60d6b2;--warn:#ffbd69}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#173653 0,var(--bg) 42%);color:var(--text);font:15px/1.55 ui-sans-serif,system-ui;padding:28px}
main{max-width:1100px;margin:auto}h1{font-size:30px;margin:0 0 6px}p{color:var(--muted)}form{display:flex;gap:10px;margin:24px 0}input{flex:1;background:#081421;color:var(--text);border:1px solid var(--line);border-radius:10px;padding:13px}button{background:var(--accent);border:0;border-radius:10px;padding:0 22px;font-weight:700;color:#072019;cursor:pointer}.grid{display:grid;grid-template-columns:1fr 2fr;gap:16px}.card{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);border-radius:14px;padding:17px;margin-bottom:16px}.metric{font-size:24px;font-weight:750}.muted{color:var(--muted)}.slot,.tag{display:inline-block;padding:3px 8px;border-radius:999px;background:#183147;margin:2px;font-size:12px}.covered{color:var(--accent)}.missing{color:var(--warn)}article{border-top:1px solid var(--line);padding:14px 0}pre{white-space:pre-wrap;color:#d5e3f2}code{color:var(--accent)}@media(max-width:760px){.grid{grid-template-columns:1fr}form{flex-direction:column}button{padding:12px}}
</style></head><body><main>
<h1>FinResearch Evidence Explorer</h1><p>本地、可复现的 Query Plan → BM25 → Evidence Pack → Retrieval Critic。页面不调用外部 API，也不生成未经引用的答案。</p>
<form id="form"><input id="q" value="示例公司A 2024年研发投入是多少？" aria-label="研究问题"><button>检索证据</button></form>
<section id="out" class="muted">等待查询…</section>
</main><script>
const out=document.querySelector('#out'),form=document.querySelector('#form'),q=document.querySelector('#q');
const el=(tag,text,cls)=>{const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n};
function card(title){const n=el('div','', 'card');n.append(el('h2',title));return n}
form.addEventListener('submit',async e=>{e.preventDefault();out.textContent='检索中…';try{const r=await fetch('/api/search',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q.value})});const d=await r.json();if(!r.ok)throw new Error(d.error||'request failed');out.textContent='';const grid=el('div','', 'grid');const left=el('div',''),right=el('div','');
let c=card('执行概览');c.append(el('div',d.plan.route,'metric'),el('p',`${d.harness_version} · ${d.latency_ms.toFixed(2)} ms · ${d.evidence.length} chunks · ${d.estimated_tokens} est. tokens`,'muted'));for(const [id,status] of d.verification.slot_coverage)c.append(el('span',`${id}: ${status}`,'slot '+status));c.append(el('p',d.verification.reason,d.verification.sufficient?'covered':'missing'));if(d.retry_gate)c.append(el('p',`Retry Gate: ${d.retry_gate.should_retry?'retry':'stop'} · ${d.retry_gate.stop_reason||d.retry_gate.query_delta}`,'muted'));left.append(c);
c=card('Query Plan');c.append(el('pre',JSON.stringify(d.plan,null,2)));left.append(c);
c=card('证据');for(const x of d.evidence){const a=el('article','');a.append(el('strong',`#${x.rank} ${x.title||x.doc_id}`),el('div',`${x.company||'未标注公司'} · ${x.year||'未标注年份'} · p.${x.page??'?'} · score ${x.score.toFixed(4)}`,'muted'),el('p',x.text));if(x.contains_untrusted_instruction)a.append(el('div','⚠ 检测到不可信指令，critic 已 fail closed','missing'));a.append(el('code',`source query: ${x.source_query}`));c.append(a)}right.append(c);grid.append(left,right);out.append(grid)}catch(err){out.textContent='错误：'+err.message}});
</script></body></html>"""


def build_demo_payload(
    question: str,
    documents: list[Document],
    retriever: BM25Retriever,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    started = time.perf_counter()
    plan = QueryAnalyzer().analyze(question)
    retrieved = []
    for query in plan.sub_queries[:4]:
        retrieved.extend((result, query, 1) for result in retriever.search(query, top_k=top_k))
    budget = ResearchBudget(max_evidence_chunks=max(top_k, 1), max_neighbor_chunks=2)
    pack = EvidencePackBuilder(documents).build(retrieved, plan.evidence_slots, budget)
    decision = RetrievalCritic().review(plan, pack)
    retry_gate = None
    if not decision.sufficient:
        retry_gate = asdict(RetrievalValueGate().decide(
            missing_slots=decision.missing_evidence,
            proposed_queries=decision.follow_up_queries or decision.missing_evidence,
            used_queries=plan.sub_queries,
            remaining_rounds=1,
        ))
    evidence = []
    for rank, item in enumerate(pack.items, start=1):
        row = asdict(item)
        row["rank"] = rank
        evidence.append(row)
    return {
        "question": question,
        "harness_version": "v3-dev-0",
        "plan": asdict(plan),
        "verification": asdict(decision),
        "evidence": evidence,
        "estimated_tokens": pack.estimated_tokens,
        "retry_gate": retry_gate,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


def _handler(documents: list[Document], top_k: int):
    retriever = BM25Retriever(documents)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/":
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            self._send(_HTML.encode(), "text/html; charset=utf-8")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/search":
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                question = str(body.get("question", "")).strip()
                if not question:
                    raise ValueError("question must not be empty")
                payload = build_demo_payload(question, documents, retriever, top_k=top_k)
                self._send(
                    json.dumps(payload, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(
                    json.dumps({"error": str(exc)}, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                    HTTPStatus.BAD_REQUEST,
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local FinResearch evidence explorer")
    parser.add_argument(
        "--documents", nargs="+", default=["data/demo/documents.jsonl"], help="JSONL corpora"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    documents = [doc for path in args.documents for doc in load_documents(Path(path))]
    server = ThreadingHTTPServer((args.host, args.port), _handler(documents, args.top_k))
    print(f"FinResearch demo: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
