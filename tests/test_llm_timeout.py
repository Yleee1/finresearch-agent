import time
import unittest

from finresearch.llm_client import APITimeoutError, _wall_clock_timeout


class WallClockTimeoutTests(unittest.TestCase):
    def test_interrupts_blocking_operation(self):
        started = time.monotonic()
        with self.assertRaises(APITimeoutError), _wall_clock_timeout(0.02):
            time.sleep(1)
        self.assertLess(time.monotonic() - started, 0.5)


if __name__ == "__main__":
    unittest.main()
