import type { TradeEvent } from "../types";

const LABELS: Record<string, string> = {
  "agent.dispatch": "派发子代理",
  "tool.invoke": "工具开始",
  "tool.result": "工具完成",
  "plan.update": "任务清单",
  "context.compressed": "上下文压缩",
  "model.fallback": "模型回退",
  "final.result": "最终回复",
  error: "异常",
};

function summarize(event: TradeEvent): string {
  const p = event.payload ?? {};
  switch (event.type) {
    case "agent.dispatch":
      return `${p.agent}：${String(p.demands ?? "").slice(0, 60)}`;
    case "tool.invoke":
      return `${p.tool}（${JSON.stringify(p.args ?? {}).slice(0, 70)}）`;
    case "tool.result": {
      if (p.circuit) return `${p.tool} 熔断状态 ${p.circuit}：${p.error ?? ""}`;
      if (p.error) return `${p.tool} 失败：${p.error}`;
      if (p.elapsed_ms !== undefined) return `${p.tool}（${p.agent ?? ""}）耗时 ${p.elapsed_ms}ms`;
      if (p.hit_count !== undefined) {
        const strategy = p.recall_strategy ? ` / ${p.recall_strategy}` : "";
        return `${p.tool} 命中 ${p.hit_count} 条${strategy}`;
      }
      if (p.order) return `${p.tool} → ${p.order.order_id} ${p.order.status}`;
      if (p.saved) return `${p.tool} 已记住：${p.saved}`;
      return String(p.tool ?? "");
    }
    case "plan.update":
      return (p.tasks ?? [])
        .map((task: any) => `${task.subject}[${task.state}]`)
        .join(" · ");
    case "context.compressed":
      return `摘要 ${p.summary_length} 字，压缩后上下文 ${p.context_messages} 条`;
    case "model.fallback":
      return `${p.from} 限流，已改用 ${p.to}（${String(p.reason ?? "").slice(0, 40)}）`;
    case "final.result":
      return String(p.text ?? "").slice(0, 60);
    case "error":
      return String(p.message ?? "");
    default:
      return JSON.stringify(p).slice(0, 80);
  }
}

export default function EventTimeline({ events }: { events: TradeEvent[] }) {
  return (
    <aside className="timeline">
      <h2>事件时间线</h2>
      {events.length === 0 && <p className="empty">发送一条购物意图后，这里会实时显示 Agent 在做什么。</p>}
      <ol>
        {events.map((event, index) => (
          <li key={index} className={`ev ${event.type.replace(".", "-")}`}>
            <div className="ev-head">
              <span className="tag">{LABELS[event.type] ?? event.type}</span>
              <time>{event.occurred_at.slice(11, 19)}</time>
            </div>
            <div className="ev-body">{summarize(event)}</div>
          </li>
        ))}
      </ol>
    </aside>
  );
}
