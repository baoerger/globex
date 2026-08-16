import type { ProductCard, TradeEvent } from "../types";

/** 从最近一次 product_search 相关的工具事件里取商品卡（工具结果 JSON 由 Agent 侧透传）。 */
function latestCards(events: TradeEvent[]): ProductCard[] {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.type !== "tool.result") continue;
    const cards = event.payload?.hits as ProductCard[] | undefined;
    if (cards && cards.length) return cards;
  }
  return [];
}

export default function ProductCards({ events }: { events: TradeEvent[] }) {
  const cards = latestCards(events);
  if (!cards.length) return null;

  return (
    <div className="cards">
      {cards.map((card) => (
        <article key={card.product_id} className="card">
          <header>
            <strong>{card.title}</strong>
            <span className="brand">
              {card.brand} · {card.origin_country}
            </span>
          </header>
          <div className="price">
            {card.price_major} {card.currency}
          </div>
          {card.landed_price && !card.landed_price.unavailable_reason && (
            <div className="landed">
              <div className="landed-total">
                到手价 {card.landed_price.landed_total_major} {card.landed_price.currency}
              </div>
              <div className="landed-detail">
                小计 {card.landed_price.subtotal_major} + 运费 {card.landed_price.freight_major} + 关税{" "}
                {card.landed_price.tariff_major}
                {card.landed_price.de_minimis_applied ? "（免税额度内）" : ""}
              </div>
            </div>
          )}
          <ul className="highlights">
            {card.highlights.map((highlight) => (
              <li key={highlight}>{highlight}</li>
            ))}
          </ul>
          <div className="skus">
            {card.skus.map((sku) => (
              <span key={sku.sku_id} className="sku">
                {sku.spec} · {sku.price_major} {sku.currency} · 库存 {sku.stock}
              </span>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}
