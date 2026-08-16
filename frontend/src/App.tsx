import { useEffect, useRef, useState } from "react";
import EventTimeline from "./components/EventTimeline";
import ProductCards from "./components/ProductCards";
import type { TradeEvent } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
const WS_BASE = API_BASE.replace(/^http/, "ws");

function loadOrCreate(key: string, prefix: string): string {
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const created = `${prefix}-${Math.random().toString(36).slice(2, 8)}`;
  localStorage.setItem(key, created);
  return created;
}

interface Turn {
  role: "buyer" | "agent";
  text: string;
}

export default function App() {
  const [sessionId] = useState(() => loadOrCreate("globex.session", "web"));
  const [buyerId] = useState(() => loadOrCreate("globex.buyer", "buyer"));
  const [events, setEvents] = useState<TradeEvent[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [streaming, setStreaming] = useState("");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WS 订阅：按会话接收 Agent 过程事件（StrictMode 下会双次挂载，用 closed 标记避免早关告警）
  useEffect(() => {
    let closed = false;
    let retryTimer: number | undefined;

    const connect = () => {
      if (closed) return;
      const ws = new WebSocket(`${WS_BASE}/commerce/events`);
      wsRef.current = ws;
      ws.onopen = () => {
        if (closed) {
          ws.close();
          return;
        }
        ws.send(JSON.stringify({ shopping_session_id: sessionId }));
        setConnected(true);
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) {
          // 断线重连，避免长任务期间丢事件
          retryTimer = window.setTimeout(connect, 1500);
        }
      };
      ws.onmessage = (message) => {
        const event: TradeEvent = JSON.parse(message.data);
        if (event.type === "token.delta") {
          setStreaming((prev) => prev + (event.payload.token ?? ""));
          return;
        }
        setEvents((prev) => [...prev, event]);
        if (event.type === "final.result") {
          setStreaming("");
          setTurns((prev) => [...prev, { role: "agent", text: event.payload.text ?? "" }]);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      wsRef.current?.close();
    };
  }, [sessionId]);

  const submit = async () => {
    const query = input.trim();
    if (!query || busy) return;
    setInput("");
    setBusy(true);
    setTurns((prev) => [...prev, { role: "buyer", text: query }]);
    try {
      await fetch(`${API_BASE}/commerce/intents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          shopping_session_id: sessionId,
          buyer_id: buyerId,
          locale: "zh-CN",
          currency: "CNY",
          raw_query: query,
        }),
      });
    } catch (error) {
      setTurns((prev) => [...prev, { role: "agent", text: `[error] 请求失败：${error}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="layout">
      <header>
        <h1>Globex 跨境购物助手</h1>
        <div className="meta">
          <span>会话 {sessionId}</span>
          <span>买家 {buyerId}</span>
          <span className={connected ? "dot on" : "dot off"}>{connected ? "事件流已连接" : "事件流断开"}</span>
        </div>
      </header>

      <main>
        <section className="chat">
          <div className="turns">
            {turns.map((turn, index) => (
              <div key={index} className={`turn ${turn.role}`}>
                <div className="who">{turn.role === "buyer" ? "我" : "Globex"}</div>
                <div className="text">{turn.text}</div>
              </div>
            ))}
            {streaming && (
              <div className="turn agent streaming">
                <div className="who">Globex</div>
                <div className="text">{streaming}</div>
              </div>
            )}
            {busy && !streaming && <div className="hint">Agent 正在处理……</div>}
          </div>

          <ProductCards events={events} />

          <div className="composer">
            <textarea
              value={input}
              placeholder="例如：我人在美国，250 美元预算买个降噪耳机寄美国，到手价多少？"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submit();
                }
              }}
            />
            <button onClick={() => void submit()} disabled={busy || !input.trim()}>
              {busy ? "处理中" : "发送"}
            </button>
          </div>
        </section>

        <EventTimeline events={events} />
      </main>
    </div>
  );
}
