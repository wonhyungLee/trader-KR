from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional, Set

import websockets

from src.brokers.kis_broker import KISBroker
from src.monitor.state_store import StateStore


MENULIST = (
    "code|time|price|prdy_sign|prdy_vrss|prdy_ctrt|wgh_avrg_prc|open|high|low|"
    "ask1|bid1|trade_vol|acml_vol|acml_tr_pbmn|sell_cnt|buy_cnt|net_buy_cnt|"
    "strength|tot_ask_qty|tot_bid_qty|trade_type|buy_ratio|prdy_vol_rate|"
    "open_time|open_sign|open_vrss|high_time|high_sign|high_vrss|"
    "low_time|low_sign|low_vrss|bsop_date|market_status|trh_stop_yn|"
    "ask_qty|bid_qty|tot_ask_qty2|tot_bid_qty2|vol_turn_rate|"
    "prev_same_time_vol|prev_same_time_ratio|time_type|rand_close_yn|vi_base_price"
)
FIELD_NAMES = MENULIST.split("|")
FIELD_COUNT = len(FIELD_NAMES)


class KISWebSocketClient:
    def __init__(
        self,
        settings: dict,
        state: StateStore,
        on_tick: Callable[[str, float, Optional[str], str], None],
    ):
        self.settings = settings
        self.state = state
        self.on_tick = on_tick
        self.broker = KISBroker(settings)
        self.url = self.broker.ws_url
        self.custtype = self.broker.custtype
        monitor = settings.get("monitor", {}) or {}
        self.tr_id = monitor.get("ws_tr_id", "H0STCNT0")
        # KIS 권고: 종목 구독/해제 메시지는 과도하게 연속 전송하지 않는다(권장: 0.2초 간격)
        self.subscribe_interval_sec = float(monitor.get("ws_subscribe_interval_sec", 0.2))
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._send_lock = asyncio.Lock()
        self._action_q: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.target_subs: Set[str] = set()

    async def _send(self, message: str):
        if not self._ws:
            return
        async with self._send_lock:
            await self._ws.send(message)

    async def subscribe(self, code: str):
        if not self._ws:
            return
        payload = {
            "header": {
                "approval_key": self._approval_key,
                "custtype": self.custtype,
                "tr_type": "1",
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": self.tr_id, "tr_key": code}},
        }
        await self._send(json.dumps(payload))
        self.state.mark_subscribed(code)

    async def unsubscribe(self, code: str):
        if not self._ws:
            return
        payload = {
            "header": {
                "approval_key": self._approval_key,
                "custtype": self.custtype,
                "tr_type": "2",
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": self.tr_id, "tr_key": code}},
        }
        await self._send(json.dumps(payload))
        self.state.mark_unsubscribed(code)

    async def set_targets(self, targets: Set[str]):
        self.target_subs = set(targets)
        current = set(self.state.current_subs)
        to_add = targets - current
        to_remove = current - targets
        for code in sorted(to_remove):
            await self._action_q.put(("unsubscribe", code))
        for code in sorted(to_add):
            await self._action_q.put(("subscribe", code))

    async def _apply_targets(self):
        current = set(self.state.current_subs)
        to_add = self.target_subs - current
        for code in sorted(to_add):
            await self._action_q.put(("subscribe", code))

    async def _send_loop(self):
        while True:
            action, code = await self._action_q.get()
            try:
                if action == "subscribe":
                    await self.subscribe(code)
                elif action == "unsubscribe":
                    await self.unsubscribe(code)
            except Exception as exc:
                logging.warning("ws send failed %s %s: %s", action, code, exc)
            finally:
                # subscribe/unsubscribe 폭주 방지
                await asyncio.sleep(max(0.2, self.subscribe_interval_sec))

    def _handle_trade_payload(self, data_cnt: int, payload: str):
        fields = payload.split("^")
        if len(fields) < 3:
            return
        # best-effort field count resolution
        if data_cnt > 0 and len(fields) % data_cnt == 0:
            field_count = len(fields) // data_cnt
        else:
            field_count = FIELD_COUNT if FIELD_COUNT > 0 else len(fields)
        for idx in range(max(1, data_cnt)):
            offset = idx * field_count
            if offset + 2 >= len(fields):
                break
            code = fields[offset].strip().zfill(6)
            try:
                price = float(fields[offset + 2].replace(",", ""))
            except Exception:
                continue
            ts = fields[offset + 1].strip() if offset + 1 < len(fields) else None
            self.on_tick(code, price, ts, "ws")

    async def _recv_loop(self):
        assert self._ws is not None
        async for message in self._ws:
            if not message:
                continue
            if message[0] in ("0", "1"):
                parts = message.split("|")
                if len(parts) < 4:
                    continue
                tr_id = parts[1]
                if tr_id != self.tr_id:
                    continue
                try:
                    data_cnt = int(parts[2])
                except Exception:
                    data_cnt = 1
                payload = parts[3]
                self._handle_trade_payload(data_cnt, payload)
            else:
                try:
                    obj = json.loads(message)
                except Exception:
                    continue
                tr_id = obj.get("header", {}).get("tr_id")
                if tr_id == "PINGPONG":
                    try:
                        # KIS는 PINGPONG 메시지 payload를 그대로 응답하는 방식을 사용한다.
                        await self._ws.send(message)
                    except Exception:
                        pass

    async def run_forever(self):
        backoff = 1
        while True:
            try:
                self._approval_key = self.broker.issue_ws_approval()
                async with websockets.connect(self.url, ping_interval=None) as ws:
                    logging.info("ws connected: %s", self.url)
                    self._ws = ws
                    # reconnect 시 stale action queue를 비워서 불필요한 unsubscribe/subcribe를 줄인다
                    self._action_q = asyncio.Queue()
                    # Reset current subs on reconnect; we will resubscribe to targets.
                    self.state.current_subs.clear()
                    await self._apply_targets()
                    send_task = asyncio.create_task(self._send_loop())
                    recv_task = asyncio.create_task(self._recv_loop())
                    done, pending = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
            except Exception as exc:
                logging.warning("ws reconnect in %ss: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            finally:
                self._ws = None
