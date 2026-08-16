/// <reference types="vite/client" />

export type TradeEventType =
  | "agent.dispatch"
  | "tool.invoke"
  | "tool.result"
  | "token.delta"
  | "plan.update"
  | "context.compressed"
  | "model.fallback"
  | "final.result"
  | "error";

export interface TradeEvent {
  type: TradeEventType;
  payload: Record<string, any>;
  occurred_at: string;
}

export interface LandedPrice {
  ship_to: string;
  subtotal_major: number;
  freight_major: number;
  tariff_major: number;
  tariff_rate: number;
  de_minimis_applied: boolean;
  landed_total_major: number;
  currency: string;
  unavailable_reason?: string;
}

export interface ProductCard {
  product_id: string;
  title: string;
  brand: string;
  category: string;
  origin_country: string;
  price_major: number;
  currency: string;
  highlights: string[];
  skus: { sku_id: string; spec: string; price_major: number; currency: string; stock: number }[];
  score: number;
  landed_price?: LandedPrice;
}
