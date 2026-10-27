"""Minimal Trading 212 API client (demo/paper by default)."""
import base64
import os
import time

import requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

BASE_URLS = {
    "demo": "https://demo.trading212.com/api/v0",
    "live": "https://live.trading212.com/api/v0",
}


class T212:
    def __init__(self):
        env = os.getenv("T212_ENV", "demo")
        if env != "demo":
            raise RuntimeError("Refusing to run against a non-demo account.")
        self.base = BASE_URLS[env]
        key, secret = os.environ["T212_API_KEY"], os.environ["T212_API_SECRET"]
        token = base64.b64encode(f"{key}:{secret}".encode()).decode()
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Basic {token}"})

    def _req(self, method, path, retries=5, **kw):
        for attempt in range(retries):
            r = self.s.request(method, self.base + path, timeout=30, **kw)
            if r.status_code == 429:  # rate limited: back off and retry
                time.sleep(int(r.headers.get("x-ratelimit-reset-after", 0)) or 2 * (attempt + 1))
                continue
            if not r.ok:
                raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
            return r.json() if r.content else None
        raise RuntimeError(f"{method} {path}: rate limited too many times")

    def cash(self):
        return self._req("GET", "/equity/account/cash")

    def account_info(self):
        return self._req("GET", "/equity/account/info")

    def positions(self):
        return self._req("GET", "/equity/portfolio")

    def pending_orders(self):
        return self._req("GET", "/equity/orders")

    def wait_for_fills(self, timeout=90):
        end = time.time() + timeout
        while time.time() < end:
            if not self.pending_orders():
                return True
            time.sleep(5)
        return False

    def instruments(self):
        return self._req("GET", "/equity/metadata/instruments")

    def market_order(self, ticker, quantity):
        """quantity > 0 buys, < 0 sells."""
        return self._req("POST", "/equity/orders/market",
                         json={"ticker": ticker, "quantity": round(quantity, 4)})
