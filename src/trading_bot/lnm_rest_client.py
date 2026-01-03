# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import hashlib
import hmac
import json as jsonlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests


@dataclass
class LNMarketsConfig:
    key: str
    secret: str
    passphrase: str
    network: str = "mainnet"  # "testnet" or "mainnet"


class LNMarketsRestClient:
    """
    v3 client with signature exactly like docs:
      prehash = timestamp + method.lower() + url.pathname + data
      data = JSON.stringify(body) OR url.search (incl leading '?', or '')

    We keep base_url including /v3, but we SIGN the pathname including /v3.
    """

    def __init__(self, cfg: LNMarketsConfig):
        self.key = cfg.key
        self.secret = cfg.secret
        self.passphrase = cfg.passphrase
        self.network = cfg.network.strip().lower()

        host = "api.lnmarkets.com" if self.network == "mainnet" else "api.testnet4.lnmarkets.com"
        self.base_url = f"https://{host}/v3"

        self.session = requests.Session()
        self.timeout_s = 20

    @staticmethod
    def _now_ms_str() -> str:
        return str(int(time.time() * 1000))

    @staticmethod
    def _compact_json(obj: Dict[str, Any]) -> str:
        # entspricht JSON.stringify ohne Whitespaces
        return jsonlib.dumps(obj, separators=(",", ":"), ensure_ascii=False)

    def _sign(self, timestamp: str, method_lower: str, path: str, data: str) -> str:
        prehash = f"{timestamp}{method_lower}{path}{data}"
        mac = hmac.new(self.secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(mac).decode("utf-8")

    def _js_number_normalize(x):
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, dict):
            return {k: _js_number_normalize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_js_number_normalize(v) for v in x]
        return x



    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        auth: bool = True,
    ):
        method_req = method.strip().upper()      # für requests
        method_sig = method.strip().lower()      # für SIGNATUR (WICHTIG!)

        if not path.startswith("/"):
            path = "/" + path

        # URL basiert bei dir auf https://host/v3
        url = self.base_url + path

        # Query
        params = params or {}
        query_string = urlencode(params, doseq=True)
        if query_string:
            url = url + "?" + query_string

        # --- SIGNATURE DATA ---
        # Doku: data ist entweder JSON-body (kompakt) ODER querystring inkl '?'
        if method_sig in ("get", "delete"):
            data_for_sig = ("?" + query_string) if query_string else ""
            body_text = None
        else:
            payload = payload or {}
            body_text = jsonlib.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            data_for_sig = body_text

        # SigPath MUSS /v3/... sein (einmal!)
        sig_path = "/v3" + path if not path.startswith("/v3/") else path

        ts = self._now_ms_str()
        sig = self._sign(ts, method_sig, sig_path, data_for_sig)

        headers = {}
        if auth:
            headers.update({
                "LNM-ACCESS-KEY": self.key,
                "LNM-ACCESS-PASSPHRASE": self.passphrase,
                "LNM-ACCESS-TIMESTAMP": ts,
                "LNM-ACCESS-SIGNATURE": sig,
            })
        if body_text is not None:
            headers["Content-Type"] = "application/json"

        resp = self.session.request(
            method=method_req,
            url=url,
            headers=headers,
            data=body_text,   # exakt den String senden, den du signierst
            timeout=self.timeout_s,
        )

        if resp.status_code >= 400:
            raise RuntimeError(
                f"HTTP {resp.status_code}\n"
                f"URL: {resp.url}\n"
                f"BodySent: {body_text}\n"
                f"SigPath: {sig_path}\n"
                f"SigData: {data_for_sig}\n"
                f"Response: {resp.text}"
            )

        return resp.json() if resp.text else None


    def get_account(self):
        return self.request("get", "/account", auth=True)

    def get_futures_ticker(self):
        return self.request("get", "/futures/ticker", auth=False)
