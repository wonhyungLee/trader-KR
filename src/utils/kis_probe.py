from __future__ import annotations

import argparse
import logging
import time

from src.brokers.kis_broker import KISBroker
from src.utils.config import load_settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=20.0, help="probe interval seconds")
    parser.add_argument("--successes", type=int, default=3, help="consecutive successes to stop")
    parser.add_argument("--code", type=str, default="005930", help="stock code for probe")
    args = parser.parse_args()

    settings = load_settings()
    broker = KISBroker(settings)
    success = 0
    while True:
        try:
            res = broker.get_current_price(args.code)
            if res:
                success += 1
                logging.info("probe success %s/%s", success, args.successes)
                if success >= args.successes:
                    return 0
            else:
                success = 0
        except Exception as exc:
            success = 0
            logging.warning("probe failed: %s", exc)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
